***

# Creating the IRSA Role for the Drift Job

`ROLE_NAME` is the name of the AWS IAM role that your drift Job’s pods will assume via IRSA (IAM Roles for Service Accounts).

You create a role (example name: `pm-drift-dvc-reader-role`) and attach a policy that grants read access to the S3 bucket/prefix where your DVC remote stores objects, then you put that role’s ARN in the ServiceAccount annotation `eks.amazonaws.com/role-arn`.

Concretely, your `drift-dvc-sa.yaml` should end up like:

```yaml
annotations:
  eks.amazonaws.com/role-arn: arn:aws:iam::<ACCOUNT_ID>:role/pm-drift-dvc-reader-role
```

The role name is arbitrary, but it should be descriptive and scoped to least-privilege (read-only to the DVC S3 prefix).

Below are two correct ways to create the IRSA role for your drift Job. The `eksctl` method is simplest for an ephemeral EKS cluster; the AWS CLI method is more explicit and works even if you don’t want `eksctl` to manage IAM objects.

This single command (in Option A) creates/updates all the moving parts: IAM role, trust relationship, attaches policy, and annotates the Kubernetes ServiceAccount.

***

## 0. Prerequisite: OIDC Provider for IRSA

Make sure your cluster has an IAM OIDC provider associated (IRSA needs it).

If you created the cluster with `eksctl`, you can usually run:

```bash
eksctl utils associate-iam-oidc-provider \
  --cluster <CLUSTER_NAME> \
  --region <AWS_REGION> \
  --approve
```

This is the “OIDC provider exists” prerequisite described in the EKS guide.

***

## Step 1: Create an IAM Policy (Least-Privilege S3 Read)

Create a file `pm-drift-dvc-s3-read.json` (edit bucket + prefix):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucketPrefix",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::YOUR_BUCKET",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["YOUR_DVC_REMOTE_PREFIX/*"]
        }
      }
    },
    {
      "Sid": "GetObjectsInPrefix",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::YOUR_BUCKET/YOUR_DVC_REMOTE_PREFIX/*"
    }
  ]
}
```

Create the policy:

```bash
aws iam create-policy \
  --policy-name pm-drift-dvc-s3-read \
  --policy-document file://pm-drift-dvc-s3-read.json
```

Save the returned Policy ARN (you’ll use it in the next commands).

***

## Option A (Recommended): `eksctl create iamserviceaccount`

### Step 2: Create the Role + ServiceAccount Binding (IRSA)

Run:

```bash
eksctl create iamserviceaccount \
  --cluster <CLUSTER_NAME> \
  --region <AWS_REGION> \
  --namespace default \
  --name drift-dvc-reader \
  --role-name pm-drift-dvc-reader-role \
  --attach-policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/pm-drift-dvc-s3-read \
  --approve
```

This matches the documented flow: `eksctl` creates the service account, creates the IAM role, attaches the policy, and annotates the service account with the role ARN.

### Step 3: Verify It Worked

Describe the ServiceAccount:

```bash
kubectl describe serviceaccount drift-dvc-reader -n default
```

You should see the `eks.amazonaws.com/role-arn` annotation.

***

## Option B (Explicit): AWS CLI + `kubectl` (Manual Trust Policy)

Use this if you want full control and don’t want `eksctl` to create IAM resources for you. The steps follow the EKS docs: policy → role trust policy → attach policy → annotate ServiceAccount.

### Step 1: Create the Kubernetes ServiceAccount

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: drift-dvc-reader
  namespace: default
EOF
```


### Step 2: Get Your Account ID and Cluster OIDC Issuer

```bash
account_id=$(aws sts get-caller-identity --query "Account" --output text)

oidc_provider=$(aws eks describe-cluster --name <CLUSTER_NAME> --region <AWS_REGION> \
  --query "cluster.identity.oidc.issuer" --output text | sed -e "s/^https:\\/\\///")
```


### Step 3: Create a Trust Policy for This ServiceAccount

Create `trust-relationship.json`:

```bash
cat > trust-relationship.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${account_id}:oidc-provider/${oidc_provider}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "${oidc_provider}:aud": "sts.amazonaws.com",
          "${oidc_provider}:sub": "system:serviceaccount:default:drift-dvc-reader"
        }
      }
    }
  ]
}
EOF
```

This trust relationship structure is the standard IRSA trust policy pattern.

### Step 4: Create the Role

```bash
aws iam create-role \
  --role-name pm-drift-dvc-reader-role \
  --assume-role-policy-document file://trust-relationship.json \
  --description "IRSA role for drift job to read DVC artifacts from S3"
```


### Step 5: Attach the S3-Read Policy

```bash
aws iam attach-role-policy \
  --role-name pm-drift-dvc-reader-role \
  --policy-arn arn:aws:iam::${account_id}:policy/pm-drift-dvc-s3-read
```


### Step 6: Annotate the ServiceAccount

```bash
kubectl annotate serviceaccount -n default drift-dvc-reader \
  eks.amazonaws.com/role-arn=arn:aws:iam::${account_id}:role/pm-drift-dvc-reader-role --overwrite
```


### Step 7: Confirm

You can confirm everything with:

```bash
aws iam get-role --role-name pm-drift-dvc-reader-role --query Role.AssumeRolePolicyDocument

aws iam list-attached-role-policies --role-name pm-drift-dvc-reader-role

kubectl describe serviceaccount drift-dvc-reader -n default
```

These commands verify the role trust policy, attached policies, and the ServiceAccount annotation.

***
