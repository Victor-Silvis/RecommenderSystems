# <b>Recommender Systems Projects</b>
A collection of various recommender systems that can be used to suggest items to users. These systems may use content-based filtering, where recommendations are based on the characteristics of items or users. Alternatively, they may leverage collaborative filtering, identifying connections between users based on implicit patterns in user preferences.

## What are they?
Recommender systems are algorithms designed to suggest relevant items to users by analyzing either the characteristics of the items or the behavior of similar users. Widely used in applications like streaming services, e-commerce, and social media, these systems enhance user experience by delivering personalized content. They generally fall into two main categories: content-based filtering, which recommends items based on their similarity to items a user has previously liked, and collaborative filtering, which finds patterns in user interactions to suggest items based on shared preferences. This repository provides a collection of various recommender system approaches to demonstrate how they can be implemented to provide relevant recommendations for users.

## Contents of Repo
This repository contains implementations of recommender systems developed over time. Currently, it includes two main types of systems, each designed to provide personalized recommendations. Below is an overview of the methodologies, implementations, and key features of these systems

### 1. K-Nearest Neighbors (KNN) Based System
This system identifies which user vectors are closest to a given user's vector using collaborative filtering. The fundamental principle of this approach is to leverage user-item interaction patterns, without relying on explicit user or item features. The system assumes that users with similar preferences are likely to rate items similarly.

Key Features:
- Collaborative Filtering:
The recommendation process is purely based on observed user rating patterns and the similarity between users, rather than explicit attributes of users or items.


