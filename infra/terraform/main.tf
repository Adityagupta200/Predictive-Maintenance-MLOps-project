terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.4.0"
}

provider "aws" {
  region = var.aws_region
}

resource "aws_security_group" "api_sg" {
  name        = "predictive-maintenance-api-sg"
  description = "Allow HTTP access to API"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "api_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.api_sg.id]
  key_name               = var.key_name

  associate_public_ip_address = true

  # CHANGED: render user_data from a template so we can inject an immutable Docker image tag
  user_data = templatefile("${path.module}/user_data.sh", {
    docker_image = var.docker_image
  })

  tags = {
    Name = "predictive-maintenance-api"
  }
}

output "api_public_ip" {
  value = aws_instance.api_server.public_ip
}
