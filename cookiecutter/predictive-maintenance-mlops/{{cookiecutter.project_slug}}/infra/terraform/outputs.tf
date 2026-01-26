output "api_server_public_ip" {
  value       = aws_instance.api_server.public_ip
  description = "Public IP of the EC2 instance running the API"
}