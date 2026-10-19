"""
Fine-tune Gemma 4 E2B (QLoRA, text-only) on the converted Discord dialogues.
Run convert_dataset.py first; it makes data/train-*.parquet and data/eval-*.parquet.

Setup on the GPU machine (RTX 3060 12 GB is enough; Unsloth lists ~8 GB for E2B):
    pip install -U unsloth        # needs a recent build for Gemma 4
    python train_gemma4_e2b.py --data data

Quick smoke test first (a few minutes):
    python train_gemma4_e2b.py --data data --max-steps 30 --max-train-rows 2000

If you hit out-of-memory: lower --max-seq-length (try 1024), keep --batch-size 1.
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/gemma4-e2b-discord")
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--max-train-rows", type=int, default=100_000,
                    help="the full set is ~7M rows; far too many for one run")
    ap.add_argument("--max-eval-rows", type=int, default=500)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs when > 0")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--keep-checkpoints", type=int, default=2,
                    help="how many checkpoints to keep on disk")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    args = ap.parse_args()

    # Unsloth must be imported before transformers/trl.
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    train_files = sorted(glob.glob(os.path.join(args.data, "train-*.parquet")))
    eval_files = sorted(glob.glob(os.path.join(args.data, "eval-*.parquet")))
    if not train_files or not eval_files:
        raise SystemExit(f"no train-*.parquet / eval-*.parquet in {args.data}; run convert_dataset.py first")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    ds = load_dataset("parquet", data_files={"train": train_files, "eval": eval_files})
    train_ds = ds["train"].shuffle(seed=3407).select(range(min(args.max_train_rows, len(ds["train"]))))
    eval_ds = ds["eval"].select(range(min(args.max_eval_rows, len(ds["eval"]))))

    def to_text(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            .removeprefix("<bos>")
            for m in batch["messages"]
        ]}

    # num_proc=1 keeps this safe on Windows.
    train_ds = train_ds.map(to_text, batched=True, num_proc=1, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_text, batched=True, num_proc=1, remove_columns=eval_ds.column_names)
    print(f"train rows {len(train_ds):,} | eval rows {len(eval_ds):,}")
    print("sample:\n", train_ds[0]["text"][:400])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=1,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=20,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            weight_decay=0.001,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.keep_checkpoints,
            output_dir=args.out,
            seed=3407,
            report_to="none",
        ),
    )

    # Learn only from assistant replies, not the user turns.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    final = os.path.join(args.out, "final-lora")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"saved LoRA adapter to {final}")


if __name__ == "__main__":
    main()
