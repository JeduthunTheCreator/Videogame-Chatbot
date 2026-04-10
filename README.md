## Videogame-Chatbot
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://react.dev/)

### Overview 
This project is an AI-powered conversational agent designed to assist users with a wide range of gaming-related queries. The chatbot acts as a virtual assistant for gamers, providing relevant information, personalised recommendations, and interactive support across topics such as videogames, industry content, and game discovery.

By combining natural language processing with external data sources, the system delivers context-aware responses and creates a more engaging, intelligent user experience.

### System Capabilities
- Personalised game recommendations based on user preferences and interaction patterns  
- Real-time game information retrieval through external API integration  
- Multimodal interaction, allowing users to query using both text and images  
- Knowledge base integration with logical reasoning and dynamic learning capabilities

### AI techniques employed
- Rule-based systems (AIML) for structured conversational handling  
- Natural Language Processing (TF-IDF, cosine similarity) for intent matching and response generation  
- Deep Learning (Convolutional Neural Networks) for image classification  
- Knowledge-based reasoning for inference and consistency checking  

## NLP and Recommendation Logic

The chatbot uses Natural Language Processing techniques to understand user queries and retrieve relevant responses. User input is preprocessed and transformed using TF-IDF vectorisation, with cosine similarity applied to match queries against a dataset of known question–answer pairs.

This allows the system to handle variations in phrasing, tolerate minor input errors, and provide contextually relevant responses based on semantic similarity rather than exact matches.

## Goal & Future Improvements

The chatbot is designed to assist users with a wide range of gaming-related queries, from videogame recommendations to general information and content discovery.

Looking ahead, I identified opportunities to move the system closer to production by enhancing the user interface to create a more polished and engaging experience. Improving usability and interaction design would make the chatbot more intuitive and appealing for end users.

Future improvements include integrating the chatbot into a larger system capable of providing real-time assistance across multiple gaming domains, such as game recommendations, live stream information, and broader gaming-related content. This would allow the chatbot to evolve into a more comprehensive and intelligent gaming assistant.

  
## Interaction of the Various Components

The chatbot is built around an AIML (Artificial Intelligence Markup Language) kernel, which handles rule-based conversation through predefined patterns and responses stored in AIML files. This provides a foundation for handling structured queries and common interactions.

To enhance flexibility, a similarity-based NLP layer is integrated using a dataset of question–answer pairs. The text data is preprocessed (lowercasing, punctuation removal, tokenisation) and transformed using a TF-IDF vectorizer. When a user submits a query, cosine similarity is used to match it against the dataset, allowing the chatbot to retrieve the most relevant response based on semantic similarity.

For real-time information, the system integrates with the **RAWG Video Games Database API**, enabling the chatbot to fetch up-to-date game data, recommendations, and related content dynamically.

The chatbot also includes a computer vision component, using a pre-trained neural network model to classify videogame images. Users can provide an image, and the system will attempt to identify the game and return the result as part of the conversation.

In addition, a lightweight knowledge base stored in a CSV file supports basic logical reasoning. This allows the chatbot to store new information provided by users, perform simple inference, and detect contradictions where applicable.

Together, these components combine rule-based logic, NLP, external data integration, and machine learning to create a modular, multi-component AI system capable of handling diverse gaming-related queries.

## Live Preview 
Check out the live video preview of the working chatbot here: 

## Conversation Logs
### General Query Handling and Contextual Responses 
The chatbot is able to handle a wide range of general gaming-related queries, demonstrating its ability to understand user intent and provide structured, informative responses. 

As shown in the examples below, the system can respond to questions about hardware optimisation, gaming history, and industry trends by generating detailed, well-organised answers. It is also capable of guiding users through interactive conversations, such as recommending games based on genre preferences and refining suggestions based on follow-up inputs.

In addition, the chatbot can provide broader gaming-related information, including community recommendations and upcoming events, showcasing its ability to combine predefined knowledge, NLP-based matching, and contextual understanding to deliver relevant and useful responses.

### Knowledge Base and Logical Reasoning
The chatbot incorporates a structured knowledge base to support logical reasoning and fact-based queries. A set of predefined facts is stored in a CSV file using a format compatible with NLTK, allowing the system to represent relationships between entities (e.g. objects and their interactions).

Before use, the knowledge base is validated to ensure consistency, preventing contradictory or invalid information from being processed. Once validated, the chatbot is able to perform basic inference, answering user queries by reasoning over the stored facts rather than relying solely on predefined responses.

As demonstrated in the examples, the system can interpret relationships and respond to queries logically, showcasing its ability to move beyond simple pattern matching and provide more intelligent, knowledge-driven interactions.

### Image Classification and Multimodal Interaction

The chatbot integrates a computer vision component, allowing it to identify videogames from user-provided images. When prompted, the system accepts an image input and uses a pre-trained convolutional neural network to classify the content and predict the corresponding game.

As demonstrated in the examples, the model is able to correctly identify games such as *Fortnite*, *Among Us*, *Minecraft*, and *God of War*, returning results in real time. This showcases the chatbot’s ability to go beyond text-based interaction and process visual data as part of the conversation.

By combining image classification with conversational interaction, the system demonstrates a multimodal approach to AI, enabling users to interact with the chatbot using both text and images.



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
<div align="center"> Made with ❤️ by Jeduthun Idemudia </div>
