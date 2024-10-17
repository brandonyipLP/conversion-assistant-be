# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from google.cloud import aiplatform
from google.oauth2 import service_account

class ActionQueryVectorDatabase(Action):
    def name(self) -> Text:
        return "action_query_vector_database"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Set up Google Cloud credentials
        credentials = service_account.Credentials.from_service_account_file(
            '../../genial-caster-438919-i8-service-account.json'
        )
        aiplatform.init(credentials=credentials, project='genial-caster-438919-i8')

        # Get the user ID from the tracker
        user_id = tracker.current_state()['sender_id']

        # Get the user's query
        user_query = tracker.latest_message['text']

        # Generate embedding for the user's query (implement this function)
        query_embedding = generate_embedding(user_query)

        # Query the vector database
        index = aiplatform.MatchingEngineIndex(index_name="customer_data_index")
        matched_items = index.find_neighbors(
            query_vector=query_embedding,
            num_neighbors=1,
            restricts={'user_id': user_id}
        )

        if matched_items:
            relevant_info = matched_items[0].id  # This would be the datapoint_id, which could be additional info
            dispatcher.utter_message(text=f"I found this relevant information for you: {relevant_info}")
        else:
            dispatcher.utter_message(text="I couldn't find any relevant information for your query.")

        return []
