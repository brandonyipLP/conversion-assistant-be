# Rasa Pro Project

This project is a Rasa Pro application for building an advanced conversational AI assistant with Vertex AI Vector Database integration.

## Setup

1. Ensure you have Python 3.7 or later installed.
2. Set up a virtual environment:

   ```bash
   python -m venv rasa_pro_env
   source rasa_pro_env/bin/activate  # On Windows, use: rasa_pro_env\Scripts\activate
   ```

3. Install Rasa Pro:

   ```bash
   pip install rasa-pro
   ```

4. Initialize the Rasa Pro project:

   ```bash
   rasa init --no-prompt
   ```

5. Configure Rasa Pro:
   Create a `.env` file in the project root and add your Rasa Pro license key:

   ```
   RASA_PRO_LICENSE=your_license_key_here
   ```

6. Update the `endpoints.yml` and `config.yml` files to use Rasa Pro features.

7. Set up Google Cloud Project:

   - Create a Google Cloud Project
   - Enable Vertex AI API
   - Create a service account and download the JSON key file

8. Install additional dependencies:

   ```bash
   pip install google-cloud-aiplatform pandas
   ```

9. Set up Vertex AI Vector Database:
   - Create a script to initialize and populate the vector database
   - Update Rasa custom actions to query the vector database

## Vertex AI Vector Database

This project uses Vertex AI Vector Database to store and query customer-specific embeddings. Each embedding is tagged with a user ID for personalized retrieval.

## Training

Train your Rasa Pro model:
