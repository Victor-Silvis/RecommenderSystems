# <b>Recommender Systems Projects</b>
A collection of various recommender systems that can be used to suggest items to users. These systems may use content-based filtering, where recommendations are based on the characteristics of items or users. Alternatively, they may leverage collaborative filtering, identifying connections between users based on implicit patterns in user preferences.

## What are they?
Recommender systems are algorithms designed to suggest relevant items to users by analyzing either the characteristics of the items or the behavior of similar users. Widely used in applications like streaming services, e-commerce, and social media, these systems enhance user experience by delivering personalized content. They generally fall into two main categories: content-based filtering, which recommends items based on their similarity to items a user has previously liked, and collaborative filtering, which finds patterns in user interactions to suggest items based on shared preferences. This repository provides a collection of various recommender system approaches to demonstrate how they can be implemented to provide relevant recommendations for users.

## Contents of Repo
This repo consists of some of the recommender systems I made over time. Currently, the repo consists of two main types of recommender systems. 

1) A <b>KNN based system</b>. This system determines which vectors are closed to the vector of the particular user. The system is based upon a collaborative filtering methodology. Meaning that in its foundation no user or item features are used to determine what items would be recommended to a particular user. This is only based upon user rating paterns, and how similar they are to other users. And by using that information, recommending the user items that other users with similar preferences have like as well. However, one system also includes the item features on top of the user-item interactions, to test whether adding this information to the vector would improve its accuracy. Finally, the system contains of two methods of predicting the rating a user would give to an item he or she has not seen before. A regression method and a classification method. Both have its positives and negatives, so both of them are compared in the main demo notebook to see what would be most suitable. 


