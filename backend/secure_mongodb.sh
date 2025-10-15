#!/bin/bash

# MongoDB Security Configuration Script

echo "Configuring MongoDB security..."

# Create admin user
mongosh admin --eval '
db.createUser({
  user: "admin",
  pwd: "your_secure_password_here",
  roles: [
    { role: "userAdminAnyDatabase", db: "admin" },
    { role: "readWriteAnyDatabase", db: "admin" }
  ]
})
'

# Create application user for mindmap database
mongosh mindmap --eval '
db.createUser({
  user: "mindmap_user",
  pwd: "your_app_password_here",
  roles: [
    { role: "readWrite", db: "mindmap" }
  ]
})
'

# Enable authentication in MongoDB config
sudo tee -a /etc/mongod.conf <<EOF

# Enable authentication
security:
  authorization: enabled
EOF

# Restart MongoDB to apply security settings
sudo systemctl restart mongod

echo "MongoDB security configured!"
echo "Use this connection string in your .env file:"
echo "mongo_db_url=mongodb://mindmap_user:your_app_password_here@localhost:27017/mindmap"