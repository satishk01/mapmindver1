#!/bin/bash

# MongoDB Installation using Docker

echo "Installing MongoDB using Docker..."

# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create MongoDB Docker Compose file
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  mongodb:
    image: mongo:7.0
    container_name: mongodb
    restart: always
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password123
      MONGO_INITDB_DATABASE: mindmap

volumes:
  mongodb_data:
EOF

# Start MongoDB container
sudo docker-compose up -d

echo "MongoDB is now running in Docker container!"
echo "Connection string for .env file:"
echo "mongo_db_url=mongodb://admin:password123@localhost:27017/mindmap?authSource=admin"

# Check container status
sudo docker-compose ps