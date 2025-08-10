#!/bin/bash
set -e

echo "=== Course Recommendation System: Docker Setup ==="

# Detect user and group IDs for proper file permissions
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Using User ID: $USER_ID, Group ID: $GROUP_ID"

# Create .env file if it doesn't exist
if [ ! -f ../.env ]; then
    echo "Creating .env file from template..."
    cp ../.env.example ../.env
    echo ""
    echo "IMPORTANT: Update .env file with your actual credentials!"
    echo "   - MySQL password"
    echo "   - Neo4j password" 
    echo "   - API keys (Cohere, Tavily)"
    echo ""
fi

# Build the application with proper user IDs
echo "Building Docker images with user ID matching..."
docker compose build --build-arg USER_ID=$USER_ID --build-arg GROUP_ID=$GROUP_ID

echo ""
echo "Docker setup complete!"
echo ""
echo "Next steps:"
echo "1. Update ../.env file with your credentials"
echo "2. Start services: docker compose up"
echo "3. Access app at: http://localhost:7860"
echo "4. Neo4j browser at: http://localhost:7474"