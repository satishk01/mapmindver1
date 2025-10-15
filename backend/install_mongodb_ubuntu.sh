#!/bin/bash

# MongoDB Installation Script for Ubuntu 20.04/22.04

echo "Installing MongoDB on Ubuntu..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y wget curl gnupg2 software-properties-common apt-transport-https ca-certificates lsb-release

# Add MongoDB GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Update package list
sudo apt update

# Install MongoDB
sudo apt install -y mongodb-org

# Create MongoDB data directory
sudo mkdir -p /data/db
sudo chown -R mongodb:mongodb /data/db

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