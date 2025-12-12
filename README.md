Set Up: Clone the repo and create the .env file with the secret API key.

Run: Execute docker-compose up --build successfully.

Interact:

Verify the /health endpoint is running.

Test the /generate endpoint with the correct API key (Success Case).

Test the /generate endpoint with an incorrect/missing key (Authentication Failure Case).

Assess the stability under concurrent load (due to the asyncio.to_thread implementation).

Confirm the API schemas via the Swagger UI (http://localhost:8000/docs).