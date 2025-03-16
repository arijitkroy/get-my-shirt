<img src="https://get-my-shirt.streamlit.app/~/+/media/4d0fc7deb631f5de30b63e7ff2f516889323d6d268c464a993ecdcc1.png"/>

# <p align=center>A Computer Vision Powered Approach</p>

## Links
🔗 [Get-My-Shirt Website](https://get-my-shirt.streamlit.app/)  
🔗 [Website Documentation](https://docs.google.com/document/d/1pUVNLvaKnK0ZBJ5hR4H6tP-Kt6ChOVKLsWBKlJjPVAU/edit?usp=sharing)  
🔗 [Get-My-Shirt Demo Video](https://youtu.be/CqCzhZlZPBo) <br/>
🔗 [Get-My-Shirt-YOLO_Model](https://colab.research.google.com/drive/1ughT0yDnSG0hD1A4nJMxJVt3OPfKWq3t?usp=sharing)

## Project Overview
"Get My Shirt" is a web application that predicts a user's T-shirt size from an uploaded image and then recommends T-shirts from a database of scraped e-commerce platforms like Meesho & Amazon. The app leverages **computer vision, artificial intelligence, deep learning, and web scraping** to provide a personalized shopping experience.



## 🛠 Tech Stack
### Frontend & UI
- **Streamlit** – Interactive web interface
- **Custom CSS** – Enhanced UI styling

### Backend & AI Models
- **YOLOv11** – T-shirt detection & boundary extraction
- **Deep Learning Models** – Face detection and pose estimation (Mediapipe)
- **Fine-Tuned Gemini Flash 1.5 AI (Google API)** – Chatbot for fashion-related queries

### Web Scraping & Data Processing
- **Browse AI** – Extraction of T-shirt data from Meesho & Amazon
- **Pandas & NumPy** – Data cleaning, transformation, and analysis
- **CSV Storage** – Stores product details for recommendations

### Deployment
- **Streamlit Cloud** – Web hosting & deployment
- **Google AI Studio (Vertex AI)** – Fine-tuned Gemini chatbot



## 🚀 How It Works
1. **Upload an Image** – Users upload a photo of themselves wearing a T-shirt (frontal view).
2. **AI-Powered Size Prediction** – The system uses a custom-trained YOLOv11 model for T-shirt detection and Mediapipe for face detection and pose estimation.
3. **Scraped T-Shirt Recommendations** – The app fetches T-shirts database scraped from Meesho & Amazon and presents recommendations.
4. **Interactive Chatbot** – A Gemini AI-powered chatbot assists users with fashion-related queries and site navigation.
5. **Seamless Shopping Experience** – Users can browse recommended T-shirts and get direct links to purchase from e-commerce platforms.



## 📊 Model Training Results
### YOLOv11 Model (T-Shirt Detection)
- **Accuracy (mAP50):** 99.5%
- **Accuracy (mAP50-95):** 95.1%

### Fine-Tuned Gemini Flash Model (Interactive Chatbot)
- Trained specifically for fashion-related queries.
- Filters out irrelevant questions by responding: *"Sorry, couldn't process that."*



## 💡 Business Implementation
### 1️⃣ Smart Retail Fitting Rooms
- Integrate the AI model into smart mirrors or kiosks in retail stores.
- Customers can scan themselves to get their ideal T-shirt size & style recommendations in-store.

### 2️⃣ API as a Service for E-Commerce (Reducing Size-Related Returns)
- Offer a paid API that e-commerce businesses can integrate to provide AI-powered size recommendations for their customers.
- Charge per API request or set up a monthly subscription model.
- Brands can pay per use or subscribe to lower return rates.

### 3️⃣ Premium AI Features (Freemium Model)
- Offer free basic predictions but charge for advanced features like **style recommendations and fabric-based suggestions**.



## 🔮 Future Enhancements
- **Integrate more e-commerce platforms** (Flipkart, Myntra, etc.)
- **Improve T-shirt size accuracy** with more training data
- **Add a recommendation engine** based on user preferences



## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.



## ✨ Contact
For any inquiries, reach out via email or open an issue on GitHub!
