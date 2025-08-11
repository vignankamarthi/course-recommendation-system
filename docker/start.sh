#!/bin/bash

# Docker startup script for Course Recommendation System
# This script ensures proper initialization for new engineers

echo "Starting Course Recommendation System..."

# Check if .env exists, if not create from example
if [ ! -f ../.env ]; then
    echo "Creating .env file from example..."
    cp ../.env.example ../.env
    echo "Please update .env with your API keys before running the application!"
fi

# Export user/group IDs for Docker
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

# Stop any existing containers
echo "Cleaning up any existing containers..."
docker compose down

# Start fresh with build
echo "Building and starting containers..."
docker compose up --build

echo "Application should be available at http://localhost:7860"
echo "Neo4j browser available at http://localhost:7474"
echo "MySQL available at localhost:3306"