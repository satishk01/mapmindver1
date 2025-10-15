#!/bin/bash

# MongoDB Installation Script for Amazon Linux 2023 (AL2023)

echo "Installing MongoDB on Amazon Linux 2023..."

# Update system packages
sudo dnf update -y

# Add MongoDB 7.0 repository for AL2023
sudo tee /etc/yum.repos.d/mongodb-org-7.0.repo <<EOF
[mongodb-org-7.0]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/amazon/2023/mongodb-org/7.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-7.0.asc
EOF

# Install MongoDB
sudo dnf install -y mongodb-org

# Create MongoDB data directory
sudo mkdir -p /data/db
sudo chown -R mongod:mongod /data/db

# Start and enable MongoDB service
sudo systemctl daemon-reload
sudo systemctl start mongod
sudo systemctl enable mongod

# Check MongoDB status
echo "Checking MongoDB status..."
sudo systemctl status mongod

# Test MongoDB connection
echo "Testing MongoDB connection..."
mongosh --eval "db.adminCommand('ismaster')"

echo "MongoDB installation completed!"
echo "MongoDB is running on port 27017"
echo "Use this connection string in your .env file:"
echo "mongo_db_url=mongodb://localhost:27017/mindmap"