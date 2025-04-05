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

## Toward a Cognitive-Inspired Hashtag Recommendation for Twitter Data Analysis

Djenouri, Youcef & Belhadi, Asma & Srivastava, Gautam & Lin, Jerry. (2022). Toward a Cognitive-Inspired Hashtag Recommendation for Twitter Data Analysis. IEEE Transactions on Computational Social Systems. PP. 1-10. 10.1109/TCSS.2022.3169838. 

Toward a Cognitive-Inspired Hashtag Recommendation for Twitter Data Analysis, published in IEEE Transactions on Computational Social Systems, investigates a knowledge-graph and deep learning based cognitive approach for hashtag recommendation. The study first performs user community detection, and then constructs knowledge graph to infer semantic relationships in the respective twitter user communities. Each community is trained with embedded deep learning model and the best model is chosen.

**Commuity detection with Knowlege graph and Deep learning for Hashtag Recommendation (CKD-HR)**
Divide and Conquer model is used to obtain subsolutions on individual knowledge graphs which when combined with each other results in a sets of shared knowledge bases (global solution). The knowldge graph represents the semantic correlations among the hashtags in the given community. The shared knowledge base combines communities if they contain similar hashtags. 

For community detection, the study uses Louvian algorithm, which optimizes correlation among user communities, to create condensed and densely-connected graphs by using hirearchial clustering model. It merges neighbouring communities to create a strongly correlated subgraphs. Then deep learning models, based on entity embedding (here bag of words), is applied. As hashtag recommendation is studied for individual communities, different models are formed out of which the best-performing model is selected.

For new users, correlation between tweets of the user and knowledge graphs and bases in the large socal network are computed. The relevant communities are identified based on both shared and individual graphs. The best model is used to recommend the hashtags for new tweets.

## AMNN: Attention-Based Multimodal Neural Network Model for Hashtag Recommendation

Q. Yang et al., "AMNN: Attention-Based Multimodal Neural Network Model for Hashtag Recommendation," in IEEE Transactions on Computational Social Systems, vol. 7, no. 3, pp. 768-779, June 2020, doi: 10.1109/TCSS.2020.2986778.

AMNN: Attention-Based Multimodal Neural Network Model for Hashtag Recommendation, published in IEEE Transactions on Computational Social Systems, proposes a hashtag recommendation network by framing the task as a sequence generation problem using sequential input data. The study is conducted on two publicly available Instagram-based datasets—HARRISON and NUS-WIDE—as well as a collection of crawled microblogs called MM-INS. The sequence-to-sequence model employs a multimodal feature extraction encoder and a coupled decoder for hashtag recommendation.

Image Feature Extraction is carried out using a hybrid neural network architecture that incorporates convolutional neural networks (CNNs)—specifically, ResNet-50 and Inception V3—followed by a Long Short-Term Memory (LSTM) module. The LSTM output forms a time-ordered sequence, which is then passed through an attention module to capture relevant and noise-free spatial features.

Text Feature Extraction is performed using a Bi-directional LSTM (BiLSTM) model, which captures contextual information from both forward and backward directions of the input sequence. The hidden states from both directions are concatenated and fed into an attention layer. Features from the image and text modules are then fused to obtain a combined multimodal representation, which is input into a Gated Recurrent Unit (GRU)-based decoder.

During training, the hashtags are encoded and input into the GRU, which learns the correlation between hashtags and multimodal content. During inference, since ground-truth hashtags are unavailable, the model removes the dependency on previous outputs. The GRU outputs a ranked list of hashtags, from which those with the highest probabilities are recommended.

## NLP-enabled Recommendation of Hashtags for Covid Based Tweets Using Hybrid BERT-LSTM Model

Jain, Kirti & Jindal, Rajni. (2024). NLP-enabled Recommendation of Hashtags for Covid based Tweets using Hybrid BERT-LSTM Model. ACM Transactions on Asian and Low-Resource Language Information Processing. 10.1145/3640812. 

**NLP-enabled Recommendation of Hashtags for Covid Based Tweets Using Hybrid BERT-LSTM Model** published in ACM Transactions on Asian and Low-Resource Language Information Processing, aims to automate hashtagging by using BELHASH, a BERT Embedding based LSTM for Hashtag Recommendation by considering it as a multilabel clasification task using MultiLabelBinarizer. The study was perform on Low-Resource language data consisting of COVID-19 Tweets.

BELHASH primarliy consists of 5 components:
1. Hashtag Encoder - One hot encoding is performed on hashtags
2. Word Tokenizer using BERT
3. Feature extarction using LSTM
4. Parts-of-Speech tagger
5. Hashtag recommender - Top-K Ranked list based on cosine similarity of all tags and new tweet feature vector.

Although, it uses a rather simple architecture combining BERT and LSTMS, it effectively recommends hashtags on low-resource dataset (unique hashtag set of 1350 post data processing to remove noisy data) with 0.72 accuracy, 0.7 Precision, 0.66 Recall and 0.67 F1-Score.

## Hashtag Recommendation for Multimodal Microblog Using Co-Attention Network

Wang, Jiawen & Huang, Haoran & Huang, Xuanjing & Gong, Yeyun. (2017). Hashtag Recommendation for Multimodal Microblog Using Co-Attention Network. 3420-3426. 10.24963/ijcai.2017/478. 

Hashtag Recommendation for Multimodal Microblog Using Co-Attention Network, published in The Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17), proposes a co-attention network incorporating visual and textual data from tweets for hashtag recommendation by formulating it as a multi-class classification problem. The model primarly consists of three components - Feature extractor, Co-attention network and Predictor.

For image feature extraction, the proposed model utilizes a pretrained VGGNet to extract information from the image after dividing it into multiple N x N grids. For textual feature extraction, word embedding is obtained and then fed into an LSTM module. Both the feature matrices are then passed through a co-attention network which sequentially generates tweet and image attention.

**Tweet-guided visual attention** is employed as the hashtags in images can be related to only localised regions and we need to filter out noises based on textual data. The textual features are pooled and fed into an image attention layer. Based on the probability obtained, new image vector representation is obtained by the weighted sum of indivual image vectors. **Image-guided textual attention** is employed similarly. In order to obtain the textual probability distribution, the model uses the newly obtained image representation to query the original textual feature matrix. The above co-attention layer can be stacked to query complex microblogs iteratively. 

The predictor composes of a single softmax classifier which is fed with the aggregated feature vector of image and text matrix representations. A ranked list of hashtags is obtained as an output.

## Long-tail Hashtag Recommendation for Micro-videos with Graph Convolutional Network

Li, Mengmeng, Gan, Tian, Liu, Meng, Cheng, Zhiyong, Yin, Jianhua, & Nie, Liqiang. (2019). Long-tail Hashtag Recommendation for Micro-videos with Graph Convolutional Network. In Proceedings of the 27th ACM International Conference on Multimedia (MM '19), pp. 509–518. https://doi.org/10.1145/3357384.3357912

This paper addresses three key challenges in hashtag recommendation for micro-videos:
1. The absence of a benchmark dataset, which the authors address by constructing and releasing a dataset containing micro-videos, associated hashtags, and user information.
2. The lack of comprehensive study on the temporal and multimodal characteristics of micro-videos.
3. The issue of hashtag sparsity and long-tail distribution, which limits the effectiveness of traditional recommendation approaches.

To tackle these issues, the authors propose a novel multi-view representation interactive embedding model combined with graph-based information propagation. They introduce a benchmark dataset called INSVIDEO, made publicly available for research.

Given the long-tail distribution of hashtags in micro-videos, the model avoids bias by constructing a knowledge graph based on hashtag correlations. Hashtag embedding propagation is implemented through stacked Graph Convolutional Network (GCN) layers. The correlation matrix used in GCN propagation is built using four types of relationships:
1. Composition (cp)
2. Super-subordination (ss)
3. Positive correlation (po)
4. Co-occurrence (co)

When multiple relations exist, the one with the highest priority (in the above order) is selected.

The multi-view representation learning component uses parallel LSTM networks followed by a Common Space Learning strategy to process multimodal inputs. User behavior modeling leverages pretrained CNNs (for visual features) and pretrained Word2Vec models (for textual features). These features are fused and passed through a fully connected network.

The final interactive embedding model is trained end-to-end and consists of:
1. A Bi-interaction layer
2. Fully connected hidden layers
3.  A final prediction layer



















