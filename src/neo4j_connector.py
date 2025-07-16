import cohere
from neo4j import GraphDatabase
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from config import neo4j_uri, neo4j_user, neo4j_password
from config import cohere_api_key

co = cohere.Client(cohere_api_key)
cohere_model = "embed-english-v3.0"


class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def store_interaction(self, user_id, education, age_group, profession, user_query, response, user_vector):
        with self.driver.session() as session:
            session.execute_write(
                self._create_interaction, user_id, education, age_group, profession, user_query, response, user_vector
            )

    @staticmethod
    def _create_interaction(tx, user_id, education, age_group, profession, user_query, response, user_vector):
        tx.run("""
            MERGE (u:User {id: $user_id})
            SET u.education = $education,
                u.age_group = $age_group,
                u.profession = $profession
            CREATE (i:Interaction {
                query: $user_query,
                response: $response,
                user_vector: $user_vector
            })
            MERGE (u)-[:MADE]->(i)
        """, user_id=user_id,
             education=education,
             age_group=age_group,
             profession=profession,
             user_query=user_query,
             response=response,
             user_vector=user_vector)

    def get_all_user_vectors(self):
     with self.driver.session() as session:
        return session.execute_read(self._get_all_user_vectors)

    @staticmethod
    def _get_all_user_vectors(tx):
      query = """
        MATCH (u:User)-[:MADE]->(i:Interaction)
        WHERE i.user_vector IS NOT NULL
        RETURN u.id AS user_id, i.query AS query, i.user_vector AS user_vector
    """
      result = tx.run(query)
      return [
        {
            "user_id": record["user_id"],
            "query": record["query"],
            "user_vector": record["user_vector"]
        }
        for record in result
      ]
    def get_user_vector(self, education, age_group, profession, user_query):
        profile_text = (
            f"User with {education} education, aged {age_group}, "
            f"working at {profession} level. Recently asked: '{user_query}'"
        )

        response = co.embed(
            texts=[profile_text],
            model=cohere_model,
            input_type="clustering"
        )
        return response.embeddings[0]

    def get_similar_users(self, user_vector, top_n=5):
      all_users = self.get_all_user_vectors()
      similarities = []

      for user in all_users:
        try:
            similarity = cosine_similarity([user_vector], [user["user_vector"]])
            similarities.append({
                "user_id": user["user_id"],
                "query": user["query"],
                "score": similarity[0][0]
            })
        except Exception as e:
            print(f"Error computing similarity: {e}")

      similarities.sort(key=lambda x: x["score"], reverse=True)
      return similarities[:top_n]

    def get_recommendations_for_user(self, query):
        with self.driver.session() as session:
            return session.execute_read(self._get_recommendations_for_user, query)

    @staticmethod
    def _get_recommendations_for_user(tx, query):
        result = tx.run("""
            MATCH (i:Interaction {query: $query})
            RETURN i.response AS response
        """, {"query": query})
        return [record["response"] for record in result]

    def get_enrolled_courses_from_similar_users(self, user_ids ):
        with self.driver.session() as session:
            return session.execute_read(self._get_enrolled_courses_from_similar_users, user_ids)

    @staticmethod
    def _get_enrolled_courses_from_similar_users(tx, user_ids):
      query = """
        MATCH (u:User)-[:ENROLLED_IN]->(c:Course)
        WHERE u.id IN $user_ids
        RETURN DISTINCT c.name AS course
     """
      result = tx.run(query, user_ids=user_ids)
      return [record["course"] for record in result]


