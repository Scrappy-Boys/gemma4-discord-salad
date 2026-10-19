# Train Gemma 4 E2B on SaladCloud (RTX 3090)

Think of it as a video game that autosaves to the cloud. Salad's machines are
people's gaming PCs, so any of them can switch off at any time. Every checkpoint is
saved to your Cloudflare R2 bucket, and when a new machine picks the job up it loads
the last save and carries on. When training finishes it saves the result and shuts
itself down so the billing stops.

## One-time setup

1. **Cloudflare R2.** Create a bucket. Create an API token with *Object Read & Write*
   for that bucket. Write down the S3 endpoint (`https://<ACCOUNT_ID>.r2.cloudflarestorage.com`),
   the Access Key ID and the Secret.
2. **SaladCloud.** Create an organization and project. Create an API key (Account, API Access).
   Write down the exact organization name and project name.
3. **Docker Hub** (or any registry) account, and Docker installed on a computer.

## Steps

**1. Build and push the image** (from this folder; the first build takes a while):

    docker build -t YOURNAME/gemma4-discord-salad:1 .
    docker push YOURNAME/gemma4-discord-salad:1

No secrets are inside the image, so a public repo is fine.

**2. Create a container group in the Salad portal:**

| Setting | Value |
|---|---|
| Name | `gemma4-discord` (lowercase, numbers, hyphens; must equal `CONTAINER_GROUP_NAME` below) |
| Image | `YOURNAME/gemma4-discord-salad:1` |
| Replicas | 1 |
| vCPU / Memory | 4 / 24 GB |
| GPU | RTX 3090 (24 GB) only |
| Storage | 40 GB or more |
| Priority | Batch or Low is cheapest. That is fine because the job auto-resumes. Pick Medium or High if interruptions annoy you. |
| Restart policy | leave the default |
| Environment variables | everything uncommented in `.env.example` (9 required) |

**3. Smoke test first (cheap).** Add `MAX_STEPS=30`, `SAMPLE_ROWS=3000`, `MAX_TRAIN_ROWS=2000` to the
environment variables and start the group. It should convert the data, train 30 steps, upload the
result and stop itself within minutes. Then, for the real run, remove those three variables and change
`R2_PREFIX` (a finished run is remembered, so the same prefix would just stop immediately).

**4. Watch it** from your own PC:

    pip install boto3
    python salad_cli.py status

(Fill in `.env` first: copy `.env.example`, same values.) Live logs are in the Salad portal.

**5. Get the result** when status says DONE:

    python salad_cli.py download

That gives you the LoRA adapter folder. Check in the portal that the container group shows as stopped.

## What protects your wallet

- Stops itself when finished (calls Salad's stop API).
- Gives up after 3 crashes (`MAX_FAILURES`) and saves the crash log to R2 under `logs/`.
- If the trainer prints nothing for 45 minutes (`STALL_MINUTES`), it asks Salad for a new machine.
- Hard stop after 36 hours total (`MAX_WALLCLOCK_HOURS`).
- Wrong GPU or too little VRAM on a node: asks for a different node.
- Bad R2 or Salad settings: stops instead of looping forever.

If the self-stop ever fails, the logs say so loudly. Stop the group by hand in the portal.

## Tuning

- **Out of memory:** set `BATCH_SIZE=2` (or 1) and `GRAD_ACCUM=8` (or 16), keeping the product at 16.
- **Checkpoint spacing:** `SAVE_STEPS=250` is a guess. Look at the log speed and pick a value that
  gives roughly one checkpoint every 20 to 30 minutes.
- **More or less data:** `MAX_TRAIN_ROWS` (default 100,000 of about 7 million).

## What is and isn't tested

Tested here against a fake S3 server and a stand-in trainer: clean run, hard kill then resume from the
exact saved step, repeated crashes, hung trainer, node with no GPU, bad config, and the status and
download tool.

**Not tested** (needs a GPU and a Docker daemon, neither available where this was written): building the
Dockerfile, the real Unsloth and Gemma 4 training, the real R2 service (it uses standard S3 calls with
settings R2 needs), and Salad's own portal. Expect to fix small things on the first smoke test, which is
what step 3 is for.
