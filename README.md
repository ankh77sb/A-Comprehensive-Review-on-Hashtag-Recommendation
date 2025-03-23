# A Comprehensive Review on Hashtag Recommendation: From Traditional to Deep Learning and Beyond
The exponential growth of user-generated content on social media platforms has precipitated significant challenges in information management, particularly in content organization, retrieval, and discovery. Hashtags, as a fundamental categorization mechanism, play a pivotal role in enhancing content visibility and user engagement. However, the development of accurate and robust hashtag recommendation systems remains a complex and evolving research challenge. Existing surveys in this domain are limited in scope and recency, focusing narrowly on specific platforms (X and Sina Weibo), methodologies, or timeframes. To address this gap, this review article conducts a systematic analysis of hashtag recommendation systems, comprehensively examining recent advancements across several dimensions. Serving as a foundational resource for researchers and practitioners, this work aims to catalyze innovation in social media content organization, thereby advancing user experience and driving innovation in social media content management and discovery.

Below are summaries of few popular research papers on Hashtag Recommendation:
## Hashtag recommendation for photo sharing services

Zhang, S., Yao, Y., Xu, F., Tong, H., Yan, X., & Lu, J. (2019). Hashtag Recommendation for Photo Sharing Services. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 5805-5812. https://doi.org/10.1609/aaai.v33i01.33015805

Hashtag Recommendation for Photo Sharing Services, published in the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19), proposes MACON model for recommending hashtags for photo sharing services such as Pinterest and Instagram based on two requirements:
1. Content Modeling Module - Hybrid modelling of the Image and Text in the post
2. User Habit Modeling Module - The modeling of users’ tagging habits based on historical posts of the user aligning to the current post.
   
**Content Modelling Module**
First both Text and Image are modelled separately by using LSTM and pre-trained VGG-16 respectively. To model Text-Image interaction, a parallel co-attention mechanism is employed, which can simultaneously learn the importance of each feature vector in both text features and image features towards the recommendation target.

**User Habit Modelling Module**
This module consists of two major steps. The first step samples a small number of users’ historical posts and the corresponding hashtags as external memory. Here, one can use user-based random sampling, community-based random sampling or user-based temporal sampling. The second step learns the tagging habits in these historical posts and connects the habits with the current post to tag using correlation score.

The output of the content modelling module (post feature vector denoted as p) and user habit modeling module (influence vector denoted as t) are concatenated to make the recommendations.






