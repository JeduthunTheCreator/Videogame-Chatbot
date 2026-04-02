## Videogame-Chatbot
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://react.dev/)

### Overview 
This project is an AI-powered conversational agent designed to interact with users on topics related to video games. The videogame chatbot serves as a virtual assistant for gamers, aiming to enhance their gaming experience by providing relevant information, suggestions and support through an interactive and user-friendly interface. 

### System Capabilities
- User Interaction
- Game Recommendations
- Game information
- Multi-platform Support
- Knowledge base and learning

### AI techniques employed
- Experts Systems
- Natural Language Processing
- Deep Learning

## Goal & Future Improvements 
  
## Interaction of the various components
The chatbot is initialised using the AIML (Artificial Intelligence Markup Language) kernel. This kernel uses a set of predefined patterns and responses (stored in AIML files) to interact with users. The program loads a dataset containing pairs of questions and answers. This dataset is used to enhance the chatbot's ability to respond to user queries by using a similarity-based approach. Text preprocessing steps, such as converting to lowercase and removing punctuation, are applied to standardize the dataset. When a user inputs a question, the program computes the similarity between the input and the questions in the dataset using TF-IDF vectorizer and cosine similsrity. This helps in finding the most relevant response. The program integrates with the RAWG Videogame Database API to fetch real-time information about videogames. This enables the chatbot to provide up-to-date game recommendations and details. The chatbot includes functionality to classify images of video games using a pre-trained neural network model. Users can ask the chatbot to identify a game by providing an image, and the chatbot responds with the game's name. The chatbot utilizes a knowledge base stored in a csv file to perform logical inference. This allows the chatbot to learn and store new information provided by the user and check for contradictions. Overall the code and files work together to create an interactive and informative chatbot that assits users with their video-game related queries, offering a blend of hardcoded responses, API-based data retrieval, image classification and logical reasoning. 

## Live Preview 
Check out the live preview of the working chatbot here: 

## Conversation Logs


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
