#!/bin/bash

# MongoDB Installation Script for Amazon Linux 2 (AL2)

echo "Installing MongoDB on Amazon Linux 2..."

# Update system packages
sudo yum update -y

# Add MongoDB 6.0 repository for AL2
sudo tee /etc/yum.repos.d/mongodb-org-6.0.repo <<EOF
[mongodb-org-6.0]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/amazon/2/mongodb-org/6.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-6.0.asc
EOF

# Install MongoDB
sudo yum install -y mongodb-org

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
mongo --eval "db.adminCommand('ismaster')"

echo "MongoDB installation completed!"
echo "MongoDB is running on port 27017"
echo "Use this connection string in your .env file:"
echo "mongo_db_url=mongodb://localhost:27017/mindmap"