# face-rec-book is a book recommendation web application using face recognition and book recommendation technology.
and be senior project for bachelor degree by Kanjana Donpraitee and Sunita Kachornvitaya.
<dl>
  <dd>The first model is for face recognition. It uses Deep Convolutional Neural Network which consists of Face Detection, Face Landmark Estimation, Encoding Faces ,and Classify model for extracts each face. Multiple algorithm including K-Nearest Neighbor (KNN), Gaussian Naive Bayes, Random Forest ,and DecisionTree. The KNN is the most appropriate algorithm to build the model that has 100% accuracy. The second model is for the book recommendation. It uses Hybrid filtering combining Content-based and Collaborative Filtering, which based on training data from Goodreads (Website for worldwide books review) to recommend books. Multiple algorithm including Singular Value Decomposition (SVD), SVD++, BaselineOnly, CoClustering, K Nearest Neighbors (KNN) Basic, KNNWithMeans, KNNWithZScore, Non-negative Matrix Factorization, SlopeOne, and NormalPredictor are tested on these models. The SVD  is the most appropriate algorithm to build the model that RMSE and MAE give the least error results of 0.9117 and 0.7288 respectively. The book recommendation application can detect and identify the face of the customer and recommend books according to the customer's interest.</dd>
</dl>

### Face Recognition 
<dl>
  <dd>using the Python Library named <b> face_recognition </b>is finding the face characteristics of each person from pictures to numbers to be able to build Classification Model for facial recognition according to the steps as shown in the picture.
  
[![1-1.png](https://i.postimg.cc/Y28cNgGX/1-1.png)](https://postimg.cc/nMswpX17)
  </dd>
</dl>

### Book Recommendation 
<dl>
  <dd>using Hybrid Filtering by combining Content-based and Collaborative Filtering as shown in the picture.
  
[![1-2.png](https://i.postimg.cc/fLy8s3CK/1-2.png)](https://postimg.cc/2bpQFSPb)

  <b> Content-based </b>  using the Python Library named <b> scikit-learn</b>  to create Vector Representations and Cosine Similarity, it uses information from the book title, author name, year of publication, language, and tags to determine the similarity of the book. 
  
  <b> Collaborative Filtering </b> using Python Library named <b> surprise</b>  to create Recommender Model for Collaborative Filtering and Model Training with 10 types of Algorithm, Singular Value Decomposition (SVD), SVD ++, BaselineOnly, CoClustering, K Nearest Neighbors (KNN) Basic, KNNWithMeans, KNNWithZScore, Non-negative Matrix Factorization, SlopeOne, and NormalPredictor. As well as evaluating the accuracy by evaluating the Root Mean Square Error (RMSE).
  
  From the work of Content-based and Collaborative Filtering combined into a <b> Hybrid Filtering </b> using Content-based To find books that are similar to the books that users have read, then use Collaborative Filtering to calculate predictions as Estimated prediction of books that are likely to be liked by users. And will bring the values ​​in the order of the top 10 books that users may like
  </dd>
</dl>

### Book recommendation web application
<dl>
  <dd>Developing a web application using <b> Flask </b>, a framework used to create a Web application developed in Python, which divides system development into two parts: face recognition via Video Streaming. And book recommendations from data and recommend model.
  
[![1-3.png](https://i.postimg.cc/GhW4F955/1-3.png)](https://postimg.cc/Wh7pjpjG)
  </dd>
</dl>

  
