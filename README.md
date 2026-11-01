# Knowledge-Driven IoT Fault Diagnosis Assistant

An AI copilot that watches simulated factory equipment telemetry, detects
abnormal readings, and explains the likely cause in plain English — grounded
in the equipment's own fault-reference documentation via RAG, with a citation,
not a guess.

Built for a college "Startup" subject project. See `CLAUDE.md` for how this
repository is built with Claude Code, and `docs/` for the full brief,
architecture, roadmap and tech-stack decisions.

```
publisher ──MQTT/TLS──▶ EMQX Cloud ──▶ subscriber ──▶ SQLite ──▶ Streamlit :8502
(simulated motor)                       │  Isolation Forest        Grafana   :3000
                                        └─ anomaly ─▶ RAG (Chroma) ─▶ Ollama LLM
```

## Run it locally

### With Docker (recommended)

Needs Docker, a filled-in `.env`, and [Ollama](https://ollama.com) running on
the host with the model pulled:

```bash
cp .env.example .env              # then fill in your EMQX Cloud credentials
ollama pull llama3.2:3b
docker compose up --build
```

The image trains the anomaly model, builds the vector store from
`data/manuals/` and downloads the embedding model at build time, so there is
nothing else to prepare. If you edit a manual or the training code, re-run with
`--build`.

This starts the simulated motor (publisher), the subscriber that stores and
scores its readings, and two dashboards:

- Streamlit (live view, anomaly list, diagnosis chat): http://localhost:8502
  (set `APP_PORT` to publish it on a different host port)
- Grafana (Motor Telemetry dashboard, login `admin` / `admin`): http://localhost:3000

The publisher injects a short high-temperature fault about 100 seconds after
it starts, so you should see an anomaly and a cited diagnosis appear shortly
after. Containers reach host Ollama through `host.docker.internal`.

### Without Docker

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env              # then fill in your EMQX Cloud credentials
python scripts/train_model.py     # -> models/isolation_forest.joblib
python scripts/ingest_manuals.py  # -> rag_store/
```

Then, in three terminals: `python -m src.mqtt_subscriber`,
`python scripts/generate_telemetry.py --fault=high_temperature --fault-after=20 --fault-count=10`,
and `streamlit run dashboard/app.py`.

### Tests

```bash
pip install pytest ruff
ruff check .
pytest -v
```

## Deploying to AWS (single EC2 instance)

The stack is four containers (Streamlit app, subscriber, publisher, Grafana)
that share one SQLite file, plus an Ollama container for the LLM. That shape
fits one EC2 instance running Docker Compose. ECS/Fargate is a poor fit because
tasks would need EFS to share the SQLite file, and SQLite's file locking is not
reliable over NFS.

The image is built on your machine, pushed to a private ECR repository, and
pulled onto the instance, which runs `docker-compose.yml` plus
`docker-compose.aws.yml` (pulls from ECR and adds an Ollama container).

**Sizing.** Use `t3.xlarge` (4 vCPU, 16 GB). Ollama runs `llama3.2:3b` on CPU,
so each diagnosis takes tens of seconds, and torch plus the LLM need several
GB of RAM; `t3.large` (8 GB) is the minimum that is likely to work. The
instance costs money for as long as it runs: stop or terminate it when you are
not demoing (see [Tear down](#tear-down)).

Commands below are PowerShell, run from the repository root on your own
machine. The EC2 section marked "on the instance" is bash.

### 0. One-time setup (you)

1. Install the [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
   (`winget install Amazon.AWSCLI`) and Docker Desktop.
2. Run `aws configure` with an IAM user's access key. Set the default region to
   `ap-south-1` (or whichever region you want; use it consistently below).
   That IAM user needs permission to manage ECR, EC2, security groups and IAM
   roles/instance profiles. `AdministratorAccess` on a personal account is the
   simplest.
3. Check you are who you think you are:

```powershell
aws sts get-caller-identity
```

```powershell
$REGION  = "ap-south-1"
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$REPO    = "iot-fault-assistant"
$REGISTRY = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
$IMAGE_URI = "${REGISTRY}/${REPO}:latest"
```

### 1. Build and push the image to ECR

```powershell
aws ecr create-repository --repository-name $REPO --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY

docker build --platform linux/amd64 -t $IMAGE_URI .
docker push $IMAGE_URI
```

`--platform linux/amd64` matters if your laptop is ARM; the instance is x86.
The `.env` file is never copied into the image.

### 2. Create the instance role (lets the instance pull from ECR)

```powershell
aws iam create-role --role-name iot-fault-assistant-ec2 --assume-role-policy-document file://deploy/ec2-trust-policy.json
aws iam attach-role-policy --role-name iot-fault-assistant-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam create-instance-profile --instance-profile-name iot-fault-assistant-ec2
aws iam add-role-to-instance-profile --instance-profile-name iot-fault-assistant-ec2 --role-name iot-fault-assistant-ec2
```

### 3. Security group and SSH key

Only your current public IP can reach SSH, Streamlit and Grafana. Streamlit has
no login of its own, so do not open 8502 to `0.0.0.0/0`.

```powershell
$MYIP = (Invoke-RestMethod https://checkip.amazonaws.com).Trim()
$VPC  = aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text --region $REGION
$SG   = aws ec2 create-security-group --group-name iot-fault-assistant --description "IoT fault assistant" --vpc-id $VPC --query GroupId --output text --region $REGION
foreach ($port in 22, 8502, 3000) {
  aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port $port --cidr "$MYIP/32" --region $REGION
}

aws ec2 create-key-pair --key-name iot-fault-assistant --query KeyMaterial --output text --region $REGION | Out-File -Encoding ascii iot-fault-assistant.pem
icacls .\iot-fault-assistant.pem /inheritance:r /grant:r "${env:USERNAME}:R"
```

Keep the `.pem` out of git (it is gitignored). If your IP changes later,
authorize the new one with the same `authorize-security-group-ingress` command.

### 4. Launch the instance

```powershell
$AMI = aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text --region $REGION

$IID = aws ec2 run-instances --image-id $AMI --instance-type t3.xlarge --key-name iot-fault-assistant `
  --security-group-ids $SG --iam-instance-profile Name=iot-fault-assistant-ec2 `
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}" `
  --metadata-options HttpTokens=required `
  --user-data file://deploy/ec2-user-data.sh `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=iot-fault-assistant}]" `
  --query "Instances[0].InstanceId" --output text --region $REGION

aws ec2 wait instance-running --instance-ids $IID --region $REGION
$IP = aws ec2 describe-instances --instance-ids $IID --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region $REGION
$IP
```

If `run-instances` fails with "Invalid IAM Instance Profile", wait ~15 seconds
(IAM is eventually consistent) and run it again. `deploy/ec2-user-data.sh`
installs Docker and the Compose plugin on first boot.

The public IP changes if you stop and start the instance. For a stable address,
allocate an Elastic IP (`aws ec2 allocate-address`, then
`aws ec2 associate-address --instance-id $IID --allocation-id <id>`); an
Elastic IP that is not attached to a running instance is billed.

### 5. Copy the config onto the instance

```powershell
$KEY = ".\iot-fault-assistant.pem"
$SSH = "-o", "StrictHostKeyChecking=accept-new", "-i", $KEY

# Waits for the user-data script (Docker install) to finish.
ssh @SSH ec2-user@$IP "cloud-init status --wait; mkdir -p app"

scp @SSH docker-compose.yml docker-compose.aws.yml .env ec2-user@${IP}:app/
scp @SSH -r grafana ec2-user@${IP}:app/

# Settings that only the instance needs. Use a real Grafana password: port 3000 is reachable from the internet.
ssh @SSH ec2-user@$IP "printf '\nIMAGE_URI=$IMAGE_URI\nGRAFANA_ADMIN_PASSWORD=change-me-please\nCOMPOSE_FILE=docker-compose.yml:docker-compose.aws.yml\n' >> app/.env"
```

`COMPOSE_FILE` in the instance's `.env` makes a plain `docker compose ...` use
both compose files. The instance's `OLLAMA_HOST` is set by
`docker-compose.aws.yml`, so the `localhost` value in `.env` is ignored there.

### 6. Start the stack (on the instance)

```powershell
ssh @SSH ec2-user@$IP
```

```bash
cd ~/app
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
docker compose up -d
docker compose logs -f ollama-pull      # first start downloads ~2 GB; wait for it to exit
docker compose ps
```

Replace `<ACCOUNT_ID>` with the 12-digit value of `$ACCOUNT`. The subscriber
waits for the model download, then the publisher starts.

### 7. Verify

From your machine (the app URL is the definition of done):

```powershell
"http://${IP}:8502"    # Streamlit
"http://${IP}:3000"    # Grafana, login admin / the password you set
```

About two minutes after the publisher starts you should see a high-temperature
anomaly on the Streamlit anomaly list and, once Ollama finishes on CPU, a
diagnosis citing the manual. If not, on the instance:

```bash
docker compose logs --tail=50 subscriber publisher app
```

### Updating a deployment

```powershell
docker build --platform linux/amd64 -t $IMAGE_URI .
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker push $IMAGE_URI
ssh @SSH ec2-user@$IP "cd app && aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY && docker compose up -d"
```

`docker-compose.aws.yml` sets `pull_policy: always`, so `up -d` fetches the new image.

### Tear down

```powershell
aws ec2 terminate-instances --instance-ids $IID --region $REGION
aws ec2 wait instance-terminated --instance-ids $IID --region $REGION
aws ec2 delete-security-group --group-id $SG --region $REGION
aws ec2 delete-key-pair --key-name iot-fault-assistant --region $REGION
aws ecr delete-repository --repository-name $REPO --force --region $REGION
aws iam remove-role-from-instance-profile --instance-profile-name iot-fault-assistant-ec2 --role-name iot-fault-assistant-ec2
aws iam delete-instance-profile --instance-profile-name iot-fault-assistant-ec2
aws iam detach-role-policy --role-name iot-fault-assistant-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam delete-role --role-name iot-fault-assistant-ec2
```

To pause instead of deleting, `aws ec2 stop-instances --instance-ids $IID`
stops the compute charge (the EBS volume is still billed); the public IP will
change on restart.

## Status

All ten roadmap phases are built. The Docker image and both compose files are
verified locally. The AWS steps above have not yet been run against a real
account; if one fails, fix this README in the same change.
