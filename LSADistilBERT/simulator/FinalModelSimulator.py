import os
import time
import aiml
import nltk
import json
import torch
import asyncio
import datetime
import threading
import subprocess
import numpy as np
import pandas as pd
import customtkinter
import tkinter as tk
from tqdm import tqdm
from tkinter import ttk
from PIL import Image, ImageTk  
from tkinter import messagebox
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from AIML_LSA import Baseline_Chatbot
from nltk.tokenize import word_tokenize
import tkinter.messagebox as messagebox
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

scroll_position = (0.0, 1.0)

class ModelSimulator(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("CRM-QA Conversational Chatbot")
        self.geometry("1300x740")
        self.resizable(False, False)

        # Load the image using PIL
        image = Image.open("C:/Users/Jude/Desktop/Thesis-1/LSADistilBERT/simulator/Image/icon_logo.png")

        # Convert the PIL Image to PhotoImage
        self.logo_image = ImageTk.PhotoImage(image)

        # Create a tkinter.Label and set the image
        self.logo_label = tk.Label(self, image=self.logo_image)
        self.logo_label.grid(row=0, column=0, sticky="nsew")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Load Images with Light and Dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Image")
        self.logo_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "C:/Users/Jude/Desktop/Thesis-1/LSADistilBERT/simulator/Image/dark_main_logo.png")), dark_image=Image.open(os.path.join(image_path, "C:/Users/Jude/Desktop/Thesis-1/LSADistilBERT/simulator/Image/light_main_logo.png")), size=(110, 43))
        self.large_logo_center = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "dark_main_logo.png")), dark_image=Image.open(os.path.join(image_path, "light_main_logo.png")), size=(200, 70))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "image_icon_light.png")), size=(20, 20))
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")), dark_image=Image.open(os.path.join(image_path, "home_light.png")), size=(20, 20))
        self.chat_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "chat_dark.png")), dark_image=Image.open(os.path.join(image_path, "chat_light.png")), size=(20, 20))
        self.parameters_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "parameters_dark.png")), dark_image=Image.open(os.path.join(image_path, "parameters_light.png")), size=(20, 20))
        self.visualization_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "visualization_dark.png")), dark_image=Image.open(os.path.join(image_path, "visualization_light.png")), size=(20, 20))
        self.about_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "information_dark.png")), dark_image=Image.open(os.path.join(image_path, "information_light.png")), size=(20, 20))
        self.send_baseline_button = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "dark_sent_img.png")), dark_image=Image.open(os.path.join(image_path, "light_sent_img.png")), size=(20, 20))
        
        # Create Navigation Frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")

        # Adjust the minimum width of the navigation_frame
        self.navigation_frame.grid_columnconfigure(0, minsize=220)  # Change 200 to your desired width
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="", image=self.logo_image, compound="left", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.navigation_frame_label.grid(row=1, column=0, padx=20, pady=20)  # Adjust the pady value to move it downward

        # Creating Buttons sidebar frame
        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=2, column=0, sticky="ew")
        self.frame_chatbot = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="QA Chatbot Model", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), image=self.chat_image, anchor="w", command=self.frame_chatbot_button_event)
        self.frame_chatbot.grid(row=5, column=0, sticky="ew")
        self.frame_visualizations = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Metrics Evaluation", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), image=self.visualization_image, anchor="w", command=self.frame_visualizations_button_event)
        self.frame_visualizations.grid(row=6, column=0, sticky="ew")
        self.frame_about_developer = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="About", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), image=self.about_image, anchor="w", command=self.frame_about_developer_button_event)
        self.frame_about_developer.grid(row=7, column=0, sticky="ew")

        # appearance mode navigation
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],  width=180, height=33, command=self.change_appearance_mode_event)
        self.appearance_mode_menu.place(x=22, y=680)  # Adjust the row value to ensure it's at the bottom

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        # home frame center logo
        self.home_frame_large_logo_center = customtkinter.CTkLabel(self.home_frame, text="", image=self.large_logo_center)
        self.home_frame_large_logo_center.grid(row=1, column=0, padx=30, pady=40)
        self.home_frame_title_center_text = customtkinter.CTkLabel(self.home_frame, text="CRM Conversational Chatbot - Question Answering", font=customtkinter.CTkFont("Bold Italic", 22))
        self.home_frame_title_center_text.grid(row=2, column=0, padx=20, pady=3)
        self.home_frame_title_center_text = customtkinter.CTkLabel(self.home_frame, text="Here is a conversational chatbot simulator that responds to user questions about the apparel industry.\n The model will include a variety of subjects, including sizes, order and \npayment transactions, founders, and several more.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.home_frame_title_center_text.grid(row=3, column=0, padx=20, pady=5)
        
        # instruction label - home frame
        self.home_frame_instructions = customtkinter.CTkLabel(self.home_frame, text="Simulator Execution", font=customtkinter.CTkFont("Bold Italic", 20))
        self.home_frame_instructions.place(x=45, y=395)
        self.home_frame_instructions_1 = customtkinter.CTkLabel(self.home_frame, text="1. DistilBERT Tokenization ", font=customtkinter.CTkFont("Bold Italic", 14))
        self.home_frame_instructions_1.place(x=45, y=425)
        self.home_frame_instructions_2 = customtkinter.CTkLabel(self.home_frame, text="2. Execute the proposed DistilBERT+LSA model for Training.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.home_frame_instructions_2.place(x=45, y=450)
        self.home_frame_instructions_3 = customtkinter.CTkLabel(self.home_frame, text="3. Test now the QA Model Chatbot for Proposed.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.home_frame_instructions_3.place(x=45, y=475)
        self.home_frame_instructions_4 = customtkinter.CTkLabel(self.home_frame, text="4. Examine the data visualization and foresee the model comparison findings.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.home_frame_instructions_4.place(x=45, y=500)
        self.home_frame_about_us = customtkinter.CTkLabel(self.home_frame, text="About Us", font=customtkinter.CTkFont("Bold Italic", 20))
        self.home_frame_about_us.place(x=700, y=395)
        self.home_frame_learn_more = customtkinter.CTkLabel(self.home_frame, text="To learn more about us, click the icon below.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.home_frame_learn_more.place(x=700, y=425)
        
        # about us frame button
        self.proposed_learnmore_button = customtkinter.CTkButton(self.home_frame, corner_radius=5, width=120, height=28, text="About Us", anchor="w")
        self.proposed_learnmore_button.place(x=700, y=455)
        self.proposed_learnmore_button.configure(command=show_about_us_message)

        # create visualization frame (baseline, proposed and system performance)
        self.visualization_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.visualization_frame.grid_columnconfigure(0, weight=1)
        self.visualization_center_text = customtkinter.CTkLabel(self.visualization_frame, text="Baseline and Proposed Model Visualization", font=customtkinter.CTkFont("Bold Italic", 22))
        self.visualization_center_text.place(x=300, y=100)

        # create visualization context
        self.visualization_context = customtkinter.CTkLabel(self.visualization_frame, text="Visualizations compare the baseline to the proposed pre-trained transformer-based model, DistilBERT. It's used for NLP \ntasks such as sentiment analysis, named entity identification, and question-answering, and can be fine-tuned \nfor specific applications with smaller datasets. DistilBERT has strong contextual \ncomprehension and can understand word connections in a statement.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.visualization_context.place(x=130, y=170)
        self.visualization_heading_context = customtkinter.CTkLabel(self.visualization_frame, text="Comparative Visualization Line Graph \n(Proposed and Baseline Model)", font=customtkinter.CTkFont("Bold Italic", 16))
        self.visualization_heading_context.place(x=380, y=290)

        # create buttons for proposed, baseline and system performance visualizations
        self.proposed_visualization_button = customtkinter.CTkButton(self.visualization_frame, corner_radius=5, width=185, height=45, text="Proposed\n(DistilBERT and LSA)")
        self.proposed_visualization_button.place(x=320, y=340)
        self.proposed_visualization_button.bind("<Button-1>", lambda e: show_proposed_model_visualization('visualization_frame', 'proposed_file_path'))

        self.baseline_visualization_button = customtkinter.CTkButton(self.visualization_frame, corner_radius=5, width=185, height=45, text="Baseline\n(AIML and LSA)")
        self.baseline_visualization_button.place(x=510, y=340)
        self.baseline_visualization_button.bind("<Button-1>", lambda e: show_baseline_model_visualization('visualization_frame', 'baseline_file_path'))

        self.baseline_metrics_results = customtkinter.CTkButton(self.visualization_frame, corner_radius=5, width=185, height=45, text="Baseline Uploaded Stats")
        self.baseline_metrics_results.place(x=510, y=392)
        self.baseline_metrics_results.bind("<Button-1>", lambda e: open_baseline_metrics_excel('visualization_frame'))

        self.proposed_metrics_results = customtkinter.CTkButton(self.visualization_frame, corner_radius=5, width=185, height=45, text="Proposed Uploaded Stats")
        self.proposed_metrics_results.place(x=320, y=392)
        self.proposed_metrics_results.bind("<Button-1>", lambda e: open_proposed_metrics_excel('visualization_frame'))

        self.system_performance_f1score_button = customtkinter.CTkButton(self.visualization_frame, corner_radius=5, width=300, height=55, text="Model F1 Score Comparison \nProposed and Baseline")
        self.system_performance_f1score_button.place(x=360, y=455)
        self.system_performance_f1score_button.bind("<Button-1>", lambda e: show_accuracy_training_comparision('visualization_frame', 'proposed_file_path', 'baseline_file_path'))

        self.system_performance_precision_button = customtkinter.CTkButton(self.visualization_frame, corner_radius=5, width=300, height=55, text="Model Precision Comparison \nProposed and Baseline")
        self.system_performance_precision_button.place(x=360, y=515)
        self.system_performance_precision_button.bind("<Button-1>", lambda e: model_precision_comparison('visualization_frame'))

        # create chatbot frame
        self.chatbot_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.chatbot_frame.grid_columnconfigure(0, weight=1)
        self.chatbot_frame_center_text = customtkinter.CTkLabel(self.chatbot_frame, text="Chatbot Conversation", font=customtkinter.CTkFont("Bold Italic", 22))
        self.chatbot_frame_center_text.place(x=420, y=80)

        # baseline components - label, text entry, textbox and button
        self.chatbot_frame_baseline_label_text = customtkinter.CTkLabel(self.chatbot_frame, text="Baseline Chatbot Model", font=customtkinter.CTkFont("Bold Italic", 17))
        self.chatbot_frame_baseline_label_text.place(x=180, y=130)

        # Get the current timestamp
        baseline_current_time = datetime.datetime.now()
        # Format the date as "Today" and time in AM/PM format
        baseline_formatted_time = baseline_current_time.strftime("%I:%M %p")
        # Create the opening greeting with current time
        baseline_opening_greeting = f"\nChatbot (Baseline) ({baseline_formatted_time}):\nHello! How can I assist you? (Type 'exit' to end the conversation)\n\n"

        # Display area for baseline
        self.chatbot_baseline_convo = customtkinter.CTkTextbox(self.chatbot_frame, width=515, height=480, font=customtkinter.CTkFont(size=14))
        self.chatbot_baseline_convo.place(x=10, y=170)
        self.chatbot_baseline_convo.insert("end", baseline_opening_greeting, "chatbot")
        self.chatbot_baseline_convo.yview_moveto(1.0)

        # Send the user input for baseline
        self.chatbot_frame_entry_baseline = customtkinter.CTkEntry(self.chatbot_frame, placeholder_text="Type your question...", width=440, height=60)
        self.chatbot_frame_entry_baseline.bind('<Return>', entry_baseline_return_key_event)
        self.chatbot_frame_entry_baseline.place(x=10, y=657)

        # Send button for baseline
        self.baseline_chatsent_button = customtkinter.CTkButton(self.chatbot_frame, corner_radius=5, width=45, height=60, text="Send", image=self.send_baseline_button, anchor="w")
        self.baseline_chatsent_button.place(x=455, y=658)
        self.baseline_chatsent_button.bind('<Return>', entry_baseline_return_key_event)
        self.baseline_chatsent_button.configure(command=lambda: send_message_baseline(baseline_chatbot_response))

        # proposed components - label, textfield and button
        self.chatbot_frame_proposed_label_text = customtkinter.CTkLabel(self.chatbot_frame, text="Proposed Chatbot Model", font=customtkinter.CTkFont("Bold Italic", 17))
        self.chatbot_frame_proposed_label_text.place(x=710, y=130)

        # Get the current timestamp
        proposed_current_time = datetime.datetime.now()
        # Format the date as "Today" and time in AM/PM format
        proposed_formatted_time = proposed_current_time.strftime("%I:%M %p")
        # Create the opening greeting with current time
        proposed_opening_greeting = f"\nChatbot (Proposed) ({proposed_formatted_time}):\nHello! How can I assist you? (Type 'exit' to end the conversation)\n\n"

        # Chatbot conversation for proposed
        self.chatbot_proposed_convo = customtkinter.CTkTextbox(self.chatbot_frame, width=530, height=480, font=customtkinter.CTkFont(size=14))
        self.chatbot_proposed_convo.place(x=540, y=170)
        self.chatbot_proposed_convo.insert("end", proposed_opening_greeting, "chatbot")

        # Send the user input for proposed
        self.chatbot_frame_entry_proposed = customtkinter.CTkEntry(self.chatbot_frame, placeholder_text="Type your question...", width=452, height=60)
        self.chatbot_frame_entry_proposed.bind('<Return>', entry_proposed_return_key_event)
        self.chatbot_frame_entry_proposed.place(x=540, y=658)
        
        # Send button for proposed
        self.proposed_chatsent_button = customtkinter.CTkButton(self.chatbot_frame, corner_radius=5, width=45, height=60, text="Send", image=self.send_baseline_button, anchor="w")
        self.proposed_chatsent_button.place(x=997, y=658)
        self.proposed_chatsent_button.bind('<Return>', entry_proposed_return_key_event)
        self.proposed_chatsent_button.configure(command=send_message_proposed)

        self.developer_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.developer_frame.grid_columnconfigure(0, weight=1)

        self.developer_frame_headline = customtkinter.CTkLabel(self.developer_frame, text="About the QA DistilBERT and LSA Chatbot", font=customtkinter.CTkFont("Bold Italic", 22))
        self.developer_frame_headline.place(x=310, y=100)

        self.developer_frame_context = customtkinter.CTkLabel(self.developer_frame, text="This simulator aims to optimize the conversational chatbot by utilizing Latent Semantic Analysis (LSA) and an enhanced Transformer-based\n model (DistilBERT) to improve customer relationship management. The problem was identified in the baseline model (E-Commerce Chatbot).\n The model needs a thorough understanding of the nuances and complexities of human language, which can result in errors in grammar, semantics,\n and sentiment analysis. Moreover, the study will employ an experimental approach following a methods-results-and-discussion structure. \nSpecifically, an iterative experiment will be conducted, involving the training and evaluation of multiple model\n versions, making adjustments to the model based on the evaluation results, and repeating this process\n until the model performs satisfactorily, achieving acceptable performance in CRM.", font=customtkinter.CTkFont("Bold Italic", 14))
        self.developer_frame_context.place(x=70, y=180)

        self.developer_frame_footer = customtkinter.CTkLabel(self.developer_frame, text="All Rights Reserved. 2023", font=customtkinter.CTkFont("Bold Italic", 13))
        self.developer_frame_footer.place(x=450, y=700)

        # select default frame
        self.select_frame_by_name("home")
        self.conversation_started = False  

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_visualizations.configure(fg_color=("gray75", "gray25") if name == "visualization" else "transparent")
        self.frame_chatbot.configure(fg_color=("gray75", "gray25") if name == "chatbot" else "transparent")
        self.frame_about_developer.configure(fg_color=("gray75", "gray25") if name == "developer" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "visualization":
            self.visualization_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.visualization_frame.grid_forget()
        if name == "chatbot":
            self.chatbot_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.chatbot_frame.grid_forget()
        if name == "developer":
            self.developer_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.developer_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_visualizations_button_event(self):
        self.select_frame_by_name("visualization")

    def frame_chatbot_button_event(self):
        self.select_frame_by_name("chatbot")

    def frame_about_developer_button_event(self):
        self.select_frame_by_name("developer")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

def entry_proposed_return_key_event(event):
    print("Enter key pushed in proposed field.")

def entry_baseline_return_key_event(event):
    print("Enter key pushed in baseline field.")

    process = subprocess.Popen(['python', 'AIML_LSA.py'], stdout=subprocess.PIPE)

    while True:
        output = process.stdout.readline()
        if not output:
            break

def show_about_us_message():
    message = "Our Application is still in progress. Stay tuned for updates!"
    messagebox.showinfo("About Us", message)

# Function to read baseline precision scores from a file
def read_precision_proposed_scores(proposed_file_path):
    precision_scores = []
    with open(r'C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Proposed Metrics Result\Proposed_Precision_Result.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "F1 score:" in line:
                precision = float(line.split(":")[1].strip())
                precision_scores.append(precision)
    return precision_scores

# Function for proposed model visualization graph
def show_proposed_model_visualization(visualization_frame, proposed_file_path):
    # Set up the initial plot
    plt.ion()  # Turn on interactive mode for plotting

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create time points from 1 to 100
    time_points = list(range(1, 702))

    # Read F1 scores for the proposed model from the file
    f1_scores_proposed = read_precision_proposed_scores(proposed_file_path)

    # Function to update the plot
    def update_plot():
        # Update the plot data
        ax.clear()  # Clear the previous plot
        ax.plot(time_points, f1_scores_proposed, label='Proposed Model', color='blue')

        # Set plot labels and title
        ax.set_ylabel('F1 Score')
        ax.set_xlabel('Items')
        ax.set_title('F1 Score Visualization of the Proposed Model Predictions')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(loc='lower right')

        # Disable window resizing
        fig.canvas.manager.window.resizable(False, False)

        # Redraw the plot
        plt.show()  # Display the plot

    # Start updating the plot
    update_plot()

proposed_file_path = r"C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Proposed Metrics Result\Proposed_Precision_Result.txt"

# Function to read baseline precision scores from a file
def read_precision_baseline_scores(file_path):
    precision_scores = []
    with open(r'C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Baseline Metrics Result\Baseline_Precision_Result.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "F1 score Score:" in line:
                precision = float(line.split(":")[1].strip())
                precision_scores.append(precision)
    return precision_scores

# Function for proposed visualization graph
def show_baseline_model_visualization(visualization_frame, file_path):
    # Set up the initial plot
    plt.ion()  # Turn on interactive mode for plotting

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create time points from 1 to 100 (adjust as needed)
    time_points = list(range(1, 702))  

    # Read precision scores from the file
    precision_scores_baseline = read_precision_baseline_scores(file_path)

    # Function to update the plot
    def update_plot():
        # Update the plot data
        ax.clear()  # Clear the previous plot
        ax.plot(time_points, precision_scores_baseline, label='Baseline Model', color='green')

        # Set plot labels and title
        ax.set_ylabel('F1 Scores')
        ax.set_xlabel('Items')
        ax.set_title('F1 Score Visualization of the Baseline Model Predictions')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(loc='lower right')

        # Disable window resizing
        fig.canvas.manager.window.resizable(False, False)

        # Redraw the plot
        plt.show()  # Display the plot

    # Start updating the plot
    update_plot()

baseline_file_path = r'C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Baseline Metrics Result\Baseline_Precision_Result.txt'

# Function for system performance visualization
def show_accuracy_training_comparision(visualization_frame, proposed_file_path, baseline_file_path):
    # Set up the initial plot
    plt.ion()  # Turn on interactive mode for plotting

    fig, ax = plt.subplots(figsize=(16, 6))

    time_points = list(range(1, 702))   # Stretch the time points list

    # Read F1 scores from files
    f1_scores_proposed = read_precision_proposed_scores(proposed_file_path)
    f1_scores_baseline = read_precision_baseline_scores(baseline_file_path)

    # Function to update the plot
    def update_plot():
        # Update the plot data
        ax.clear()  # Clear the previous plot
        ax.plot(time_points, f1_scores_proposed, label='Proposed Model', color='blue')
        ax.plot(time_points, f1_scores_baseline, label='Baseline Model', color='green')

        # Set plot labels and title
        ax.set_xlabel('Items')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Scores of the Proposed and Baseline Model Predictions')

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(loc='lower right')

        # Disable window resizing
        fig.canvas.manager.window.resizable(False, False)

        # Redraw the plot
        plt.show()  # Display the plot

    # Start updating the plot
    update_plot()

def model_precision_comparison(visualization_frame):
    # Precision scores for the baseline and proposed models
    baseline_precision = 0.97
    proposed_precision = 0.89

    # Labels for the models
    models = ["AIML + Latent Semantic Analysis", "DistilBERT + Latent Semantic Analysis"]

    # Precision scores
    precision_scores = [baseline_precision, proposed_precision]

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.bar(models, precision_scores, color=['blue', 'green'])
    
    # Add padding for x-axis label and y-axis label
    ax.set_xlabel("Question Answering Models", labelpad=10)  # You can adjust the labelpad value
    ax.set_ylabel("Precision Score %", labelpad=15)  # You can adjust the labelpad value
    ax.set_title("Overall Model Precision Comparison")

    # Add precision scores above the bars
    for i in range(len(models)):
        ax.text(models[i], precision_scores[i], f"{precision_scores[i]}%", ha='center', va='bottom')

    plt.show()

def display_initial_message():
    chatbot_instance = Baseline_Chatbot()
    return chatbot_instance  # Return the chatbot_instance

# Function to save results to Excel
def save_baseline_metrics(f1, em, cosine_similarity):
    # Define the Excel file path
    excel_file_path = "Baseline Metrics Result/baseline_metrics.xlsx"

    # Check if the file doesn't exist, then create it
    if not os.path.exists(excel_file_path):
        df = pd.DataFrame(columns=["F1 Score", "Exact Match", "Cosine Similarity"])
    else:
        df = pd.read_excel(excel_file_path, index_col=0)

    # Create a new DataFrame for the current results
    new_data = pd.DataFrame({"F1 Score": [f1], "Exact Match": [em], "Cosine Similarity": [cosine_similarity]})

    # Concatenate the current results with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

    # Save the updated DataFrame to the Excel file
    df.to_excel(excel_file_path, index=True)
    return False  # Failed to save results

def scroll_chat_to_bottom():
    app.chatbot_baseline_convo.yview_moveto(1)
    app.after(100, scroll_chat_to_bottom)

    # Place this line after you create the app object
    app.after(100, scroll_chat_to_bottom)

def display_chatbot_response(chatbot_response, user_input, formatted_time, start_time, threshold):
    global exact_matches, true_positives
    processing_time = time.time() - start_time  # Calculate the execution time in seconds

    aiml_response = chatbot_response.get_response(user_input)

    if aiml_response:
        # Display AIML response with response time
        aiml_message = f"AIML Result ({formatted_time}):\n{aiml_response}\nExecution Time: {processing_time:.2f} seconds"
        app.chatbot_baseline_convo.insert("end", aiml_message, "chatbot")
        app.update_idletasks()
    else:
        # Get LSA-based response using chatbot_response instance
        start_time = time.time()  # Update start_time here for LSA-based response
        most_similar_text, cosine_similarity_score = chatbot_response.get_chatbot_response(user_input)
        processing_time = time.time() - start_time

        # Rest of your LSA-based response code
        if cosine_similarity_score > threshold:
            chatbot_message = f"AIML+LSA Chatbot ({formatted_time}):\n{most_similar_text}\nExecution Time: {processing_time:.2f} seconds"
            app.chatbot_baseline_convo.insert("end", chatbot_message, "chatbot")
            app.update_idletasks()

            # Calculate Precision, Recall, and F1 Score using sklearn metrics
            precision = precision_score([1], [1 if cosine_similarity_score > threshold else 0])
            f1 = f1_score([1], [1 if cosine_similarity_score > threshold else 0])

            em = 1 if f1 == 1 else 0

            save_baseline_metrics(f1, em, cosine_similarity_score)
            app.update_idletasks()
        else:
            chatbot_message = f"Chatbot ({formatted_time}):\nI couldn't find an answer to your question.\nExecution Time: {processing_time:.2f} seconds"
            app.chatbot_baseline_convo.insert("end", chatbot_message, "chatbot")

    app.chatbot_baseline_convo.configure(state=customtkinter.DISABLED)
    app.update_idletasks()
    new_scroll_position = (scroll_position[0], scroll_position[1] + 1.0)

    def scroll_to_bottom():
        app.chatbot_baseline_convo.yview_moveto(new_scroll_position[1])

    app.after(50, scroll_to_bottom)

    app.chatbot_frame_entry_baseline.delete(0, "end")

def send_message_baseline(chatbot_response):
    # Initialize variables to keep track of metrics
    global exact_matches, true_positives  # Declare as global variables
    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Set the threshold for cosine similarity classification
    threshold = 0.10

    user_input = app.chatbot_frame_entry_baseline.get()

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%I:%M %p")

    # Check for ending responses
    add_ending_responses = {
        "goodbye": f"Chatbot ({formatted_time}):\nGoodbye! Have a great day.",
        "thanks": f"Chatbot ({formatted_time}):\nYou're welcome! If you have more questions, feel free to ask.",
        "see you later": f"Chatbot ({formatted_time}):\nSee you later! Take care.",
    }

    if user_input.lower() in add_ending_responses:
        app.chatbot_baseline_convo.configure(state=customtkinter.NORMAL)
        app.chatbot_baseline_convo.insert("end\n", add_ending_responses[user_input.lower()], "chatbot")
        app.update_idletasks()
        app.chatbot_baseline_convo.configure(state=customtkinter.DISABLED)
        app.chatbot_frame_entry_baseline.delete(0, "end")
        return  # Exit the function

    user_message = f"\nUser ({formatted_time}):\n{user_input}\n\n"
    app.chatbot_baseline_convo.configure(state=customtkinter.NORMAL)
    app.chatbot_baseline_convo.insert("end", user_message, "user")
    app.chatbot_baseline_convo.tag_config("user", justify="right")
    app.update_idletasks()

    # Capture the start time when the user hits enter
    start_time = time.time()

    # Call display_chatbot_response directly without a delay
    display_chatbot_response(chatbot_response, user_input, formatted_time, start_time, threshold)

    app.chatbot_frame_entry_baseline.delete(0, "end")

def entry_baseline_return_key_event(event):
    send_message_baseline(baseline_chatbot_response)

def entry_proposed_return_key_event(event):
    send_message_proposed()

# Function to save results to Excel
def save_proposed_metrics(f1, em, precision):
    # Define the Excel file path
    excel_file_path = r"C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Proposed Metrics Result\proposed_metrics.xlsx"

    # Check if the file doesn't exist, then create it
    if not os.path.exists(excel_file_path):
        df = pd.DataFrame(columns=["F1 Score", "Exact Match", "Cosine Similarity"])
    else:
        df = pd.read_excel(excel_file_path, index_col=0)

    # Create a new DataFrame for the current results
    new_data = pd.DataFrame({"F1 Score": [f1], "Exact Match": [em], "Precision": [precision]})

    # Concatenate the current results with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

    # Save the updated DataFrame to the Excel file
    df.to_excel(excel_file_path, index=True)
    return False  # Failed to save results

def calculate_metrics(true_labels, predicted_labels):
    # F1 Score
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    precision = np.sum((true_labels == 1) & (predicted_labels == 1)) / (np.sum(predicted_labels == 1) + 1e-9)
    recall = np.sum((true_labels == 1) & (predicted_labels == 1)) / (np.sum(true_labels == 1) + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    # Exact Match (EM)
    em = np.mean(true_labels == predicted_labels)

    # Precision
    precision = np.sum((true_labels == 1) & (predicted_labels == 1)) / (np.sum(predicted_labels == 1) + 1e-9)

    return f1, em, precision

processing_in_progress = False

def send_message_proposed():
        global processing_in_progress

        if processing_in_progress:
            messagebox.showinfo("Processing Request", "Another process is already in progress. Please wait...")
            return

        processing_in_progress = False

        # Create a progress bar
        progress_bar = messagebox.showinfo("Execution", "Your request is being processed. Please wait...", icon='info')    
        progress_bar = customtkinter.CTkProgressBar(app, mode='indeterminate', width=529)
        progress_bar.place(x=760 , y=640) # Adjust the coordinates as needed
        progress_bar.start()

        def background_task():
            try:
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%I:%M %p")

                # Record the start time
                start_time = time.time()

                #-----------Initialize HuggingFace Transformer Model---------------
                model_name = 'distilbert-base-uncased-distilled-squad'
                distilbert_model = DistilBertModel.from_pretrained(model_name)
                tokenizer = DistilBertTokenizer.from_pretrained(model_name)

                try:
                    # Load the training and testing data
                    with open("LSADistilBERT/simulator/train.json", "r") as f:
                        train_data = json.load(f)
                    with open("LSADistilBERT/simulator/test.json", "r") as f:
                        test_data = json.load(f)
                except FileNotFoundError:
                    root = tk.Tk()
                    root.withdraw()  # Hide the main window
                    messagebox.showwarning("File Not Found", "JSON files for training and testing cannot be located.\n \nPlease do preprocessing beforehand so that you may use the converse approach.")

                # -----------DistilBERT---------------
                # Define the function to extract embeddings using DistilBERT
                def get_distilbert_embeddings(inputs):
                    # Tokenize the inputs using the tokenizer
                    input_ids = inputs["input_ids"]
                    encoded_inputs = [tokenizer.convert_tokens_to_string(ids) for ids in input_ids]
                    encoded_inputs = tokenizer.encode_plus(encoded_inputs, padding=True, truncation=True, return_tensors="pt")
                    
                    # Forward pass through DistilBERT
                    outputs = distilbert_model(**encoded_inputs)
                    # Extract the embeddings
                    embeddings = outputs.last_hidden_state
                    return embeddings

                # -----------Embedding Extraction---------------
                # Convert tokenized questions to the required format
                train_questions = []
                for example in train_data:
                    question_tokens = example["question"]
                    question_text = tokenizer.convert_tokens_to_string([str(token) for token in question_tokens])
                    train_questions.append(question_text)

                train_inputs = {"input_ids": train_questions}

                test_questions = []
                for example in test_data:
                    question_tokens = example["question"]
                    question_text = tokenizer.convert_tokens_to_string([str(token) for token in question_tokens])
                    test_questions.append(question_text)

                test_inputs = {"input_ids": test_questions}

                # Extract the embeddings for training and testing data using DistilBERT
                train_embeddings = get_distilbert_embeddings(train_inputs)
                test_embeddings = get_distilbert_embeddings(test_inputs)

                # Check the shape of the embeddings
                print("Shape of train_embeddings:", train_embeddings.shape)
                print("Shape of test_embeddings:", test_embeddings.shape)

                # -----------Stack Embedding---------------
                # Flatten the embeddings
                train_embeddings_flat = train_embeddings.reshape(train_embeddings.shape[0] * train_embeddings.shape[1], -1)
                test_embeddings_flat = test_embeddings.reshape(test_embeddings.shape[0] * test_embeddings.shape[1], -1)

                # Detach the tensors from the computation graph and convert to NumPy arrays
                train_embeddings_flat_np = train_embeddings_flat.detach().numpy()
                test_embeddings_flat_np = test_embeddings_flat.detach().numpy()

                # Stack Embedding
                num_features = 100
                svd = TruncatedSVD(n_components=num_features)

                # Fit the SVD model to the training embeddings
                train_embeddings_transformed = svd.fit_transform(train_embeddings_flat_np)
                test_embeddings_transformed = svd.transform(test_embeddings_flat_np)

                #-----------LSA (Document Term Matrix)---------------
                # Define the LSA model to be used
                lsa = TruncatedSVD(n_components=6)

                # Fit the LSA model to the transformed training embeddings
                train_embeddings_lsa = lsa.fit_transform(train_embeddings_transformed)
                test_embeddings_lsa = lsa.transform(test_embeddings_transformed)

                #-----------LSA (SVD)---------------
                train_data_text = []
                for example in train_data:
                    context_text = example["context"]
                    question_text = example["question"]
                    answer_text = context_text[example["start_position"]:example["end_position"]+1]  # Extract the answer text dynamically
                    train_data_text.append({
                        "context": context_text,
                        "question": question_text,
                        "answer": answer_text
                    })

                # Convert token IDs to text for testing data
                test_data_text = []
                for example, embedding in zip(test_data, test_embeddings_lsa):
                    context_text = example["context"]
                    question_text = example["question"]
                    answer_text = context_text[example["start_position"]:example["end_position"]+1]  # Extract the answer text dynamically
                    test_data_text.append({
                        "context": context_text,
                        "question": question_text,
                        "answer": answer_text
                    })

                # Save the text-based train, validation, and test sets to JSON files
                with open("train_text.json", "w") as f:
                    json.dump(train_data_text, f)
                with open("test_text.json", "w") as f:
                    json.dump(test_data_text, f)

                #-----------Parameters---------------
                model_args = QuestionAnsweringArgs()
                model_args.train_batch_size = 32
                model_args.n_best_size = 30
                model_args.num_train_epochs = 8
                model_args.weight_decay = 0.05
                model_args.dropout = 0.3 

                train_args = {
                    "reprocess_input_data": False,
                    "overwrite_output_dir": False,
                    "use_cached_eval_features": False,
                    "evaluate_during_training": True,
                    "max_seq_length": 256,
                    "num_train_epochs": 8,
                    "evaluate_during_training_steps": 2000, 
                    "wandb_project": "DISTILBERT_CHAT",
                    "wandb_kwargs": {"name": f"updated_{model_name}"},
                    "save_model_every_epoch": True,
                    "save_eval_checkpoints": True,
                    "n_best_size": 30,
                    "initial_learning_rate": 0.00005,  
                    "weight_decay": 0.05, 
                    "train_batch_size": 32,  
                    "eval_batch_size": 16, 
                    "gradient_accumulation_steps": 2,
                    "max_grad_norm": 1.0,
                    "data_augmentation_techniques": ["random_masking", "random_permutation", "back_translation"],
                }

                # Create the learning rate scheduler
                num_training_steps = len(train_data) * model_args.num_train_epochs // model_args.train_batch_size
                warmup_proportion = 0.1
                warmup_steps = int(num_training_steps * warmup_proportion)
                decay_steps = num_training_steps // 5
                decay_rate = 0.01
                optimizer = torch.optim.AdamW(distilbert_model.parameters(), lr=train_args["initial_learning_rate"], weight_decay=train_args["weight_decay"])
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

                #-----------Create the QuestionAnsweringModel---------------
                model = QuestionAnsweringModel(
                    model_type='distilbert',
                    model_name=model_name,
                    args=model_args,
                    use_cuda=False,
                )

                # Load the text-based training data
                with open("train_text.json", "r") as f:
                    train_text_data = json.load(f)

                # Convert the text-based training data to DataFrame format
                train_df = pd.DataFrame(train_text_data)

                # Convert the DataFrame to the required format for training
                train_input = []
                for i, row in train_df.iterrows():
                    context = row['context']
                    question = row['question']
                    answer = row['answer']
                    
                    # Convert context and answer to string if they are lists
                    if isinstance(context, list):
                        context = ' '.join(str(item) for item in context)
                    if isinstance(answer, list):
                        answer = ' '.join(str(item) for item in answer)
                    
                    train_input.append({
                        'context': context,
                        'qas': [
                            {
                                'id': str(i),
                                'question': question,
                                'answers': [
                                    {
                                        'text': answer,
                                        'answer_start': context.find(answer),
                                    }
                                ],
                                'is_impossible': False
                            }
                        ]
                    })

                # Load the text-based test data
                with open("test_text.json", "r") as f:
                    test_text_data = json.load(f)

                # Convert the text-based test data to DataFrame format
                test_df = pd.DataFrame(test_text_data)

                # Convert the DataFrame to the required format for prediction
                test_input = []
                for i, row in test_df.iterrows():
                    context = row['context']
                    question = row['question']
                    answer = row['answer']
                    
                    # Convert context and answer to string if they are lists
                    if isinstance(context, list):
                        context = ' '.join(str(item) for item in context)
                    if isinstance(answer, list):
                        answer = ' '.join(str(item) for item in answer)
                    
                    test_input.append({
                        'context': context,
                        'qas': [
                            {
                                'id': str(i),
                                'question': question,
                                'answers': [
                                    {
                                        'text': answer,
                                        'answer_start': context.find(answer),
                                    }
                                ]
                            }
                        ]
                    })
                    
                while True:
                    user_input = app.chatbot_frame_entry_proposed.get()  # Get user input from the entry box

                    if user_input.lower() == "exit":
                        # Update the conversation textbox
                        app.chatbot_proposed_convo.insert(tk.END, "User: Goodbye! I'm happy to serve you!\n")
                        return  # Exit the function if the user wants to exit
                
                    context1 = ("OXGN is a fashion brand designed for all. From cool styles, affordable essentials and one of a kind collaboration pieces, our customers are sure to find that signature OXGN brand of style. With timeless classics from our GENERATIONS collection, the leveled-up sports inspired pieces from PREMIUM THREADS and our inclusive co-gender line COED, our brand serves to make looking cool as easy as breathing.")
                    context2 = ("GOLDEN ABC, Inc. (GABC) is a multi-awarded international fashion enterprise that is home to top proprietary brands shaping the retail industry today. Produced, marketed, and retailed under a fast-growing, dynamic family of well-differentiated, proprietary brands: PENSHOPPE, OXYGEN, FORME, MEMO, REGATTA, and BOCU. Originating in the Philippines in 1986, GABC now has more than 1000 strategically located sites and presence in different countries around Asia.")
                    context3 = ("The OXGN VIP Loyalty Card is not applicable online and it has been already been discontinued in our physical stores last December 31, 2021.")
                    context4 = ("A confirmation e-mail/ SMS will be sent for all successful orders made. This contains your order number, the order amount, item(s) ordered, your billing and delivery address, including the payment method used and how you will be receiving order updates.")
                    context5 = ("Yes, you have the option cancel your order within two (2) hours from the time it was created. You can see this on your order confirmation email or on your online store order history. Once the time frame has passed, the option to cancel will no longer appear on your end. If in case you still need to cancel your order but you no longer have the option to do so, you can send an email to shop@oxgnfashion.com and our Customer Care Team will get in touch with you.")
                    context6 = ("We cannot include paper bags for Online Orders shipped to their chosen delivery address but we can definitely provide for those Online orders that were checked-out on a Store Pick-up option.")
                    context7 = ("After you tap on REVIEW MY CART, you will see the option to send your order as a Gift. Sending it as a gift means that a gift receipt will be used instead of a regular sales invoice. A gift receipt shows proof of purchase but leaves out the amount spent.")
                    context8 = ("Once the check out process has been completed, changes to the order in terms of item/size/color/quantity can no longer be made. What you can do is cancel your order completely within two (2) hours from the time you placed it so you can place a new one with the changes you want.")
                    context9 = ("The first step is to click 'Add to Cart' once you've found something you want to save the items in your order basket. Second, click the 'Review My Cart' button to be redirected to the cart page, where you can make any final changes to your order, such as quantity, size, or style. Third, address Finder will appear on your screen. Please enter your delivery address by using the dropdown menus. If you accidentally exit out before finishing your details, click on 'Address Finder' to bring it back up. This is to prevent your orders from being held due to incorrect or incomplete shipping information.")
                    context10 = ("Yes, you have the option cancel your order within two (2) hours from the time it was created either it is shipped to your prefered address or pick-up in your chosen You can view this on your order confirmation email or on your online store order history. Once the time frame has passed, the option to cancel will no longer appear on your end.")
                    context11 = ("All unclaimed packages beyond the scheduled pick-up date are considered canceled and will be put back to our stocks.")
                    context12 = ("You can now order online and pick up the items in your chosen OXGN branches: SM CITY NORTH EDSA, SM CITY SAN LAZARO, AYALA TRINOMA, ROBINSONS GALLERIA, SM MEGAMALL, SM CITY CEBU ,AYALA CENTER CEBU, MARKET MARKET, SM CITY FAIRVIEW, SM SEASIDE CEBU, SM LANANG, GMALL TAGUM, SM CITY DAVAO, GAISANO MALL OF DAVAO, SM CITY EAST ORTIGAS, SM MARIKINA, AYALA GLORIETTA, SM CITY MANILA, SM CITY GRAND CENTRAL, MALL OF ASIA, SM CITY SUCAT, ROBINSONS LAOAG, SM CITY BAGUIO, SM CITY PAMPANGA, SM CITY CLARK, SM CITY DASMARIÑAS, SM CALAMBA, SM CITY LUCENA, SM CITY LIPA, SM CITY ILO-ILO, SM CITY BACOLOD, SM Roxas, AYALA CENTRAL BLOC, Robinsons Tacloban, MINDPRO, ROBINSONS ILIGAN, Gaisano Mall Digos, SM CITY GENERAL SANTOS, KCC MALL MARBEL, SM CITY CAUAYAN, ISABELA, SM CITY MARILAO, TUTUBAN CENTER, Val Alabang, AYALA MALLS MANILA BAY, SM CITY LEGAZPI, ALTURAS MALL TAGBILARAN, SM PREMIER, SM BUTUAN, Mall of Alnor, SM City Tuguegarao, Ever Commonwelath, ROBINSONS BACOLOD, Fishermall.")
                    context13 = ("As much as we'd like to, this option is not yet allowed at the moment.")
                    context14 = ("This feature gives you the option to buy the items online and pick it up in our stores.")
                    context15 = ("Oxgnfashion.com accepts the following forms of payment: GCash, Credit/Debit Card and COD (Cash-on-Delivery). During the order checkout process, you will be asked to select the payment method of your choice. If you select Credit/Debit Card, you will be redirected to our payment portal where you will be asked to enter your card details.")
                    context16 = ("No, there is no option to save your credit or debit card information on your online store profile. Even if you have already used the same card, we do not store your card details to protect the security of your data.")
                    context17 = ("Customers will receive an e-mail notification when their order is read for pick up. It is stated in the notification that they can pick up the order within 7 days from the receipt of e-mail.")
                    context18 = ("No worries! You may assign a representative that may pick-up your order and have them present the following: Representative's Valid ID, Authorization letter with your Full Name and Signature, Photocopy of your Valid ID, and Screenshot of Order No. from email.")
                    context19 = ("If you have placed order via our online store and chose to pick it up in one of branches, please prepare the following to be checked by our store partners: Valid ID, Screenshot of Order No. from email.")
                    context20 = ("Customers with prepaid orders are allowed to request for up to 7 days extension from the first pick-up schedule.")
                    context21 = ("If your order gets returned to our warehouse, we're sorry to let you know that it is already considered as canceled from our end. This is only done after two (2) unsuccessful delivery attempts.")
                    context22 = ("Our delivery partners will make two (2) attempts in getting your package delivered. If they missed you on the first instance, the courier will either send you an SMS or give you a call to let you know that he came.")
                    context23 = ("As we believe in making the latest fashion styles accessible to everyone, we can assure you that we have wide reach across the Philippines. However, due to geographical and logistical restrictions, there might be some isolated areas we cannot serve but rest assured we'll do our best to get to your location.")
                    context24 = ("Customers who need to contact our shipping partners can do so using the following methods: For LBC, customers can send an email to customercare@lbcexpress.com or call (02) 8 858 59 99. For Go21, customers can visit the Inquiry Page at https://booking.go21.com.ph/ or call +63 2 832 99 27. For J&T, customers can visit https://www.jtexpress.sg/contact-us or call (02) 911 1888. For WSI, customers can send an email to communicationsupport@wsi.ph or call (02) 8533 3888. For feedback on our shipping partners or delivery concerns, customers can send an email to shop@oxgnfashion.com or fill out the Website Contact Us Form.")
                    context25 = ("Once your order has been handed over to our shipping partners, you will receive an email notification along with your order tracking number.")
                    context26 = ("Customers who wish to return items purchased from our online store can do so by packing the item along with the sales invoice and bringing it to any Penshoppe store nationwide. The return must be made within 30 days from the date of purchase, with all tags attached and the item unused and unwashed.")
                    context27 = ("If you receive a gift item from our online store and want to have it exchanged for a different size/color/style, you have the option to do so following our Returns Policy.")
                    context28 = ("If you need to process a return or exchange transaction and have lost your online store receipt, please reach out to us with the following information: your order number, the item you will be returning, the reason for the return, the boutique you will be going to, and the date.")
                    context29 = ("No, our warehouse does not accept returns from customers but our physical stores do.")
                    context30 = ("In the event that you received a damaged or incorrect item, it is important to reach out to us immediately so that we can investigate the situation.")
                    context31 = ("Refund requests are always subject to validation. It is important to note that a change in size, style, or color is not eligible for a refund.")
                    context32 = ("Creating an account on the OXGN Online Store is easy and allows you to keep track of your orders through the Order History feature.")
                    context33 = ("Yes, you can still shop without making an online store account! Just place the items you want in your cart and proceed to check out.")
                    context34 = ("If you have forgotten your password for your online store account, don't worry as it is an easy process to reset it.")
                    context35 = ("First time to sign in? Tap on the account icon so you can start adding your account details.")
                    context36 = ("Reach out to +639399101047 or complete the form below and we’ll gladly assist you.")
                    context37 = ("These Terms and Conditions ('Terms', 'Terms and Conditions') govern your relationship with OXGN Website (the 'Service') operated by Golden ABC, Incorporated ('us', 'we', or 'our').")
                    context38 = ("There are ten available variations of tops for men: Regular Fit Shirt, Resort Fit Shirt, Long Sleeve, Hoodie, Relax set-in Pullover, Easy Fit, Jersy, Slim Fit, Oversized, and Tank Top.")
                    context39 = ("There are five available variations of bottoms for men: Skinny, Tapered (Slim Fit, Woven, Knit), and Jogger Pants.")
                    context40 = ("There are two available variations of loungewear for men: Robe and Lounge Set.")
                    context41 = ("There are six available variations of footwear for men: Sliders, Single Band, Velcro Sliders, Regular Fliptops, Lace-up, and Lace-up Runner.")
                    context42 = ("There are seven available variations of tops for ladies: Regular, Boxy, Slim fit Tee, Shirt (Short Sleeve), Sports Bra, Hoodie, and Regular Pullover.")
                    context43 = ("There are two available variations of dress for ladies: Oversized Dress, Long Sleeve Regular Fit.")
                    context44 = ("There are ten available variations of bottoms for ladies: Skinny Jeans (Numerical,  Alpha), Wide Leg Jeans, Shorts (Numerical, Alpha), Biker Shorts (Alpha), Leggings (With Pockets, Without Pockets), Track Pants, and Urban Tapered Trousers.")
                    context45 = ("There are four available variations of footwear for ladies: Sliders, Single Band Sliders, Velcro Sliders, and Lace-up Sneakers.")
                    context46 = ("There are three available categories of accessories in total: Bags, Caps and Hats, and Wallet and Purses.")
                    context47 = ("There are four available variations of Bags in accessories: Drawstring Backpack, Bum Bag, Sling Bag, and Tote Bag.")
                    context48 = ("There are two available variations of Caps/Hat in accessories: Trucker Hat and Bucket Hat.")
                    context49 = ("There are three available variations of walltes/purses in accessories: Coin Purse (Horizontal & Vertical) and Midsized Bifold Wallet.")
                    context50 = ("For men's top regular fit shirt in double extra small, Chest: 37, Body Length: 26, Sleeve Length: 7.75, USA: 10, EU: 44, and UK: 34.")
                    context51 = ("For men's top regular fit shirt in extra small, Chest: 39, Body Length: 27, Sleeve Length: 8, USA: 12, EU: 46, and UK: 36.")
                    context52 = ("For men's top regular fit shirt in small, Chest: 41, Body Length: 28, Sleeve Length: 8.25, USA: 14, EU: 48, and UK: 38.")
                    context53 = ("For men's top regular fit shirt in medium, Chest: 43, Body Length: 29, Sleeve Length: 8.5, USA: 16, EU: 50, and UK: 40.")
                    context54 = ("For men's top regular fit shirt in large, Chest: 45, Body Length: 30, Sleeve Length: 8.75, USA: 18, EU: 52, and UK: 42.")
                    context55 = ("For men's top regular fit shirt in extra large, Chest: 47, Body Length: 31, Sleeve Length: 9, USA: 20, EU: 54, and UK: 44.")
                    context56 = ("For men's top regular fit shirt in double extra large, Chest: 49, Body Length: 32, Sleeve Length: 9.25, USA: 22, EU: 56, and UK: 46.")
                    context57 = ("For men's top resort fit shirt in double extra small, Chest: 39, Body Length: 27.5, Sleeve Length: 9.25, USA: 10, EU: 44, and UK: 34.")
                    context58 = ("For men's top resort fit shirt in extra small, Chest: 41, Body Length: 28, Sleeve Length: 9.5, USA: 12, EU: 46, and UK: 36.")
                    context59 = ("For men's top resort fit shirt in small, Chest: 43, Body Length: 28.5, Sleeve Length: 9.75, USA: 14, EU: 48, and UK: 38.")
                    context60 = ("For men's top resort fit shirt in medium, Chest: 45, Body Length: 29, Sleeve Length: 10, USA: 16, EU: 50, and UK: 40.")
                    context61 = ("For men's top resort fit shirt in large, Chest: 47, Body Length: 29.5, Sleeve Length: 10.25, USA: 18, EU: 52, and UK: 42.")
                    context62 = ("For men's top resort fit shirt in extra large, Chest: 49, Body Length: 30, Sleeve Length: 10.5, USA: 20, EU: 54, and UK: 44.")
                    context63 = ("For men's top resort fit shirt in double extra large, Chest: 51, Body Length: 30.5, Sleeve Length: 10.75, USA: 22, EU: 56, and UK: 46.")
                    context64 = ("For men's top long sleeve shirt in double extra small, Chest: 36, Body Length: 26, Sleeve Length: 24.75, USA: 10, EU: 44, and UK: 34.")
                    context65 = ("For men's top long sleeve shirt in extra small, Chest: 38, Body Length: 27, Sleeve Length: 25, USA: 12, EU: 46, and UK: 36.")
                    context66 = ("For men's top long sleeve shirt in small, Chest: 40, Body Length: 28, Sleeve Length: 25.25, USA: 14, EU: 48, and UK: 38.")
                    context67 = ("For men's top long sleeve shirt in medium, Chest: 42, Body Length: 29, Sleeve Length: 25.5, USA: 16, EU: 50, and UK: 40.")
                    context68 = ("For men's top long sleeve shirt in large, Chest: 44, Body Length: 30, Sleeve Length: 25.75, USA: 18, EU: 52, and UK: 42.")
                    context69 = ("For men's top long sleeve shirt in extra large, Chest: 46, Body Length: 31, Sleeve Length: 26, USA: 20, EU: 54, and UK: 44.")
                    context70 = ("For men's top long sleeve shirt in double extra large, Chest: 48, Body Length: 32, Sleeve Length: 26.25, USA: 22, EU: 56, and UK: 46.")
                    context71 = ("For men's top long sleeve shirt in triple extra large, Chest: 50, Body Length: 33, Sleeve Length: 26.5, USA: 24, EU: 58, and UK: 48.")
                    context72 = ("For men's top hoodie dropped shoulder shirt in double extra small, Chest: 44, Body Length: 25.5, Sleeve Length: 17.25, USA: 10, EU: 44, and UK: 34.")
                    context73 = ("For men's top hoodie dropped shoulder shirt in extra small, Chest: 46, Body Length: 26, Sleeve Length: 17.5, USA: 12, EU: 46, and UK: 36.")
                    context74 = ("For men's top hoodie dropped shoulder shirt in small, Chest: 48, Body Length: 26.5, Sleeve Length: 17.75, USA: 14, EU: 48, and UK: 38.")
                    context75 = ("For men's top hoodie dropped shoulder shirt in medium, Chest: 50, Body Length: 27, Sleeve Length: 18, USA: 16, EU: 50, and UK: 40.")
                    context76 = ("For men's top hoodie dropped shoulder shirt in large, Chest: 52, Body Length: 27.5, Sleeve Length: 18.25, USA: 18, EU: 52, and UK: 42.")
                    context77 = ("For men's top hoodie dropped shoulder shirt in extra large, Chest: 54, Body Length: 28, Sleeve Length: 18.5, USA: 20, EU: 54, and UK: 44.")
                    context78 = ("For men's top hoodie dropped shoulder shirt in double extra large, Chest: 56, Body Length: 28.5, Sleeve Length: 18.75, USA: 22, EU: 56, and UK: 46.")
                    context79 = ("For men's top hoodie dropped shoulder shirt in triple extra large, Chest: 58, Body Length: 29, Sleeve Length: 19, USA: 24, EU: 58, and UK: 48.")
                    context80 = ("For men's top hoodie set-in shoulder shirt in double extra small, Chest: 38, Body Length: 25, Sleeve Length: 24.25, USA: 10, EU: 44, and UK: 34.")
                    context81 = ("For men's top hoodie set-in shoulder shirt in extra small, Chest: 40, Body Length: 26, Sleeve Length: 24.5, USA: 12, EU: 46, and UK: 36.")
                    context82 = ("For men's top hoodie set-in shoulder shirt in small, Chest: 42, Body Length: 27, Sleeve Length: 24.75, USA: 14, EU: 48, and UK: 38.")
                    context83 = ("For men's top hoodie set-in shoulder shirt in medium, Chest: 44, Body Length: 28, Sleeve Length: 25, USA: 16, EU: 50, and UK: 40.")
                    context84 = ("For men's top hoodie set-in shoulder shirt in large, Chest: 46, Body Length: 29, Sleeve Length: 25.25, USA: 18, EU: 52, and UK: 42.")
                    context85 = ("For men's top hoodie set-in shoulder shirt in extra large, Chest: 48, Body Length: 30, Sleeve Length: 25.5, USA: 20, EU: 54, and UK: 44.")
                    context86 = ("TFor men's top hoodie set-in shoulder shirt in double extra large, Chest: 50, Body Length: 31, Sleeve Length: 25.75, USA: 22, EU: 56, and UK: 46.")
                    context87 = ("For men's top hoodie set-in shoulder shirt in triple extra large, Chest: 52, Body Length: 32, Sleeve Length: 26, USA: 24, EU: 58, and UK: 48.")
                    context88 = ("For men's top relax set-in pullover shirt in double extra small, Chest: 19, Body Length: 26, Sleeve Length: 24.25, USA: 10, EU: 44, and UK: 34.")
                    context89 = ("For men's top relax set-in pullover shirt in extra small, Chest: 20, Body Length: 27, Sleeve Length: 24.5, USA: 12, EU: 46, and UK: 36.")
                    context90 = ("For men's top relax set-in pullover shirt in small, Chest: 21, Body Length: 27, Sleeve Length: 24.75, USA: 14, EU: 48, and UK: 38.")
                    context91 = ("For men's top relax set-in pullover shirt in medium, Chest: 22, Body Length: 28, Sleeve Length: 25, USA: 16, EU: 50, and UK: 40.")
                    context92 = ("For men's top relax set-in pullover shirt in large, Chest: 23, Body Length: 29, Sleeve Length: 25.25, USA: 18, EU: 52, and UK: 42.")
                    context93 = ("For men's top relax set-in pullover shirt in extra large, Chest: 24, Body Length: 30, Sleeve Length: 25.5, USA: 20, EU: 54, and UK: 44.")
                    context94 = ("For men's top relax set-in pullover shirt in double extra large, Chest: 25, Body Length: 31, Sleeve Length: 25.75, USA: 22, EU: 56, and UK: 46.")
                    context95 = ("For men's tee easy fit shirt in double extra small, Chest: 18, Body Length: 26, Sleeve Length: 7.25, USA: 10, EU: 44, and UK: 34.")
                    context96 = ("For men's tee easy fit shirt in extra small, Chest: 19, Body Length: 26, Sleeve Length: 7.5, USA: 12, EU: 46, and UK: 36.")
                    context97 = ("For men's tee easy fit shirt in small, Chest: 20, Body Length: 27, Sleeve Length: 7.75, USA: 14, EU: 48, and UK: 38.")
                    context98 = ("For men's tee easy fit shirt in medium, Chest: 21, Body Length: 28, Sleeve Length: 8, USA: 16, EU: 50, and UK: 40.")
                    context99 = ("For men's tee easy fit shirt in large, Chest: 22, Body Length: 29, Sleeve Length: 8.25, USA: 18, EU: 52, and UK: 42.")
                    context100 = ("For men's tee easy fit shirt in double extra large, Chest: 24, Body Length: 31, Sleeve Length: 8.75, USA: 22, EU: 56, and UK: 46.")
                    context101 = ("For men's top jersey with raglan sleeves shirt in double extra small, Chest: 36, Body Length: 25, Sleeve Length: 13.5, USA: 10, EU: 44, and UK: 34.")
                    context102 = ("For men's top jersey with raglan sleeves shirt in extra small, Chest: 38, Body Length: 26, Sleeve Length: 14, USA: 12, EU: 46, and UK: 36.")
                    context103 = ("For men's top jersey with raglan sleeves shirt in small, Chest: 40, Body Length: 27, Sleeve Length: 14.5, USA: 14, EU: 48, and UK: 38.")
                    context104 = ("For men's top jersey with raglan sleeves shirt in medium, Chest: 42, Body Length: 28, Sleeve Length: 15, USA: 16, EU: 50, and UK: 40.")
                    context105 = ("For men's top jersey with raglan sleeves shirt in large, Chest: 44, Body Length: 29, Sleeve Length: 15.5, USA: 18, EU: 52, and UK: 42.")
                    context106 = ("For men's top jersey with raglan sleeves shirt in extra large, Chest: 46, Body Length: 30, Sleeve Length: 16, USA: 20, EU: 54, and UK: 44.")
                    context107 = ("For men's top jersey with raglan sleeves shirt in double extra large, Chest: 48, Body Length: 31, Sleeve Length: 16.5, USA: 22, EU: 56, and UK: 46.")
                    context108 = ("For men's top jersey with raglan sleeves shirt in triple extra large, Chest: 50, Body Length: 32, Sleeve Length: 17, USA: 24, EU: 58, and UK: 48.")
                    context109 = ("For men's tee slim fit shirt in double extra small, Chest: 17, Body Length: 25.5, Sleeve Length: 6.75, USA: 10, EU: 44, and UK: 34.")
                    context110 = ("For men's tee slim fit shirt in extra small, Chest: 18, Body Length: 26, Sleeve Length: 7, USA: 12, EU: 46, and UK: 36.")
                    context111 = ("For men's tee slim fit shirt in small, Chest: 19, Body Length: 26.5, Sleeve Length: 7.25, USA: 14, EU: 48, and UK: 38.")
                    context112 = ("For men's tee slim fit shirt in medium, Chest: 20, Body Length: 27, Sleeve Length: 7.5, USA: 16, EU: 50, and UK: 40.")
                    context113 = ("For men's tee slim fit shirt in large, Chest: 21, Body Length: 27.5, Sleeve Length: 7.75, USA: 18, EU: 52, and UK: 42.")
                    context114 = ("For men's tee slim fit shirt in extra large, Chest: 22, Body Length: 28, Sleeve Length: 8, USA: 20, EU: 54, and UK: 44.")
                    context115 = ("For men's tee slim fit shirt in double extra large, Chest: 23, Body Length: 28.5, Sleeve Length: 8.25, USA: 22, EU: 56, and UK: 46.")
                    context116 = ("For men's tee oversized fit shirt in double extra small, Chest: 19, Body Length: 27, Sleeve Length: 8.5, USA: 10, EU: 44, and UK: 34.")
                    context117 = ("For men's tee oversized fit shirt in extra small, Chest: 20, Body Length: 28, Sleeve Length: 8.75, USA: 12, EU: 46, and UK: 36.")
                    context118 = ("For men's tee oversized fit shirt in small, Chest: 21, Body Length: 29, Sleeve Length: 9, USA: 14, EU: 48, and UK: 38.")
                    context119 = ("For men's tee oversized fit shirt in medium, Chest: 22, Body Length: 30, Sleeve Length: 9.25, USA: 16, EU: 50, and UK: 40.")
                    context120 = ("For men's tee oversized fit shirt in large, Chest: 23, Body Length: 31, Sleeve Length: 9.5, USA: 18, EU: 52, and UK: 42.")
                    context121 = ("For men's tee oversized fit shirt in extra large, Chest: 24, Body Length: 32, Sleeve Length: 9.75, USA: 20, EU: 54, and UK: 44.")
                    context122 = ("For men's tee oversized fit shirt in double extra large, Chest: 25, Body Length: 33, Sleeve Length: 10, USA: 22, EU: 56, and UK: 46.")
                    context123 = ("For men's top tank top shirt in double extra small, Chest: 36, Body Length: 26, Sleeve Length: N/A, USA: 10, EU: 44, and UK: 34.")
                    context124 = ("For men's top tank top shirt in extra small, Chest: 38, Body Length: 27, Sleeve Length: N/A, USA: 12, EU: 46, and UK: 36.")
                    context125 = ("For men's top tank top shirt in small, Chest: 40, Body Length: 28, Sleeve Length: N/A, USA: 14, EU: 48, and UK: 38.")
                    context126 = ("For men's top tank top shirt in medium, Chest: 42, Body Length: 29, Sleeve Length: N/A, USA: 16, EU: 50, and UK: 40.")
                    context127 = ("For men's top tank top shirt in large, Chest: 44, Body Length: 30, Sleeve Length: N/A, USA: 18, EU: 52, and UK: 42.")
                    context128 = ("For men's top tank top shirt in double extra large, Chest: 48, Body Length: 32, Sleeve Length: N/A, USA: 22, EU: 56, and UK: 46.")
                    context129 = ("For men's jeans skinny fit (regular waist band) in size 29, Waist: 28, Hips: 34, Inseam: 28, USA: 38, EU: 38, and UK: 30.")
                    context130 = ("For men's jeans skinny fit (regular waist band) in size 30, Waist: 29, Hips: 35, Inseam: 28, USA: 39, EU: 39, and UK: 30-31.")
                    context131 = ("For men's jeans skinny fit (regular waist band) in size 31, Waist: 30, Hips: 36, Inseam: 28, USA: 40, EU: 40, and UK: 31.")
                    context132 = ("For men's jeans skinny fit (regular waist band) in size 32, Waist: 31, Hips: 37, Inseam: 29, USA: 42, EU: 42, and UK: 32.")
                    context133 = ("For men's jeans skinny fit (regular waist band) in size 34, Waist: 31, Hips: 37, Inseam: 29, USA: 42, EU: 42, and UK: 32.")
                    context134 = ("For men's jeans slim fit (regular waist band) in size 29, Waist: 27, Hips: 36, Inseam: 28, USA: 38, EU: 38, and UK: 30.")
                    context135 = ("For men's jeans slim fit (regular waist band) in size 30, Waist: 28, Hips: 37, Inseam: 28, USA: 39, EU: 39, and UK: 30-31.")
                    context136 = ("For men's jeans slim fit (regular waist band) in size 31, Waist: 29, Hips: 38, Inseam: 28, USA: 40, EU: 40, and UK: 31.")
                    context137 = ("For men's jeans slim fit (regular waist band) in size 32, Waist: 30, Hips: 39, Inseam: 29, USA: 42, EU: 42, and UK: 32.")
                    context138 = ("For men's jeans slim fit (regular waist band) in size 34, Waist: 32, Hips: 41, Inseam: 29, USA: 44, EU: 44, and UK: 34.")
                    context139 = ("For men's jeans jogger pants (garterized waist band) in size extra small, Waist: 28, Hips: 38, Inseam: 25, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context140 = ("For men's jeans jogger pants (garterized waist band) in size small, Waist: 30, Hips: 40, Inseam: 25, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context141 = ("For men's jeans jogger pants (garterized waist band) in size medium, Waist: 32, Hips: 42, Inseam: 25, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context142 = ("For men's jeans jogger pants (garterized waist band) in size large, Waist: 34, Hips: 44, Inseam: 25, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context143 = ("For men's jeans jogger pants (garterized waist band) in size extra large, Waist: 36, Hips: 46, Inseam: 25, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context144 = ("For men's tapered pants woven (regular waist band) in size extra small, Waist: 24, Hips: 37, Inseam: 25, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context145 = ("For men's tapered pants woven (regular waist band) in size small, Waist: 26, Hips: 39, Inseam: 25, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context146 = ("For men's tapered pants woven (regular waist band) in size medium, Waist: 28, Hips: 41, Inseam: 25, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context147 = ("For men's tapered pants woven (regular waist band) in size large, Waist: 30, Hips: 43, Inseam: 25, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context148 = ("For men's tapered pants woven (regular waist band) in size extra large, Waist: 32, Hips: 45, Inseam: 25, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context149 = ("For men's tapered pants knit (garterized waist band) in size extra small, Waist: 26, Hips: 38, Inseam: 26, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context150 = ("For men's tapered pants knit (garterized waist band) in size small, Waist: 28, Hips: 40, Inseam: 26, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context151 = ("For men's tapered pants knit (garterized waist band) in size medium, Waist: 30, Hips: 42, Inseam: 26, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context152 = ("For men's tapered pants knit (garterized waist band) in size large, Waist: 32, Hips: 44, Inseam: 26, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context153 = ("For men's tapered pants knit (garterized waist band) in size extra large, Waist: 34, Hips: 46, Inseam: 26, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context154 = ("For men's tapered pants with cuff-knit (garterized waist band) in size double extra small, Waist: 23, Hips: 36, Inseam: 25, USA: 32-34, EU: 32-34, and UK: 27-28.")
                    context155 = ("For men's tapered pants with cuff-knit (garterized waist band) in size extra small, Waist: 25, Hips: 38, Inseam: 25, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context156 = ("For men's tapered pants with cuff-knit (garterized waist band) in size small, Waist: 27, Hips: 40, Inseam: 25, USA: 39-40, EU: 39-40 and UK: 30-31.")
                    context157 = ("For men's tapered pants with cuff-knit (garterized waist band) in size medium, Waist: 29, Hips: 42, Inseam: 25, USA: 43-44, EU: 43-44 and UK: 31-32.")
                    context158 = ("For men's tapered pants with cuff-knit (garterized waist band) in size large, Waist: 31, Hips: 44, Inseam: 25, USA: 43-44, EU: 43-44 and UK: 33-34.")
                    context159 = ("For men's tapered pants with cuff-knit (garterized waist band) in size extra large, Waist: 33, Hips: 46, Inseam: 25, USA: 45-46, EU: 45-46 and UK: 35-36.")
                    context160 = ("For men's tapered pants with cuff-knit (garterized waist band) in size double extra large, Waist: 35, Hips: 48, Inseam: 25, USA: 48-50, EU: 48-50 and UK: 37-38.")
                    context161 = ("For men's urban shorts - knit & woven (garterized waist band) in size extra small, Waist: 28, Hips: 40, Inseam: 5, USA: 36-38, EU: 36-38 and UK: 29-30.")
                    context162 = ("For men's urban shorts - knit & woven (garterized waist band) in size small, Waist: 30, Hips: 42, Inseam: 5, USA: 39-40, EU: 39-40 and UK: 30-31.")
                    context163 = ("For men's urban shorts - knit & woven (garterized waist band) in size medium, Waist: 32, Hips: 44, Inseam: 5, USA: 40-42, EU: 40-42 and UK: 31-32.")
                    context164 = ("For men's urban shorts - knit & woven (garterized waist band) in size large, Waist: 34, Hips: 46, Inseam: 5, USA: 43-44, EU: 43-44 and UK: 33-34.")
                    context165 = ("For men's urban shorts - knit & woven (garterized waist band) in size extra large, Waist: 36, Hips: 48, Inseam: 5, USA: 45-46, EU: 45-46 and UK: 35-36.")
                    context166 = ("For men's chino shorts - woven (regular waist band) in size 28, Waist: 28, Hips: 38, Inseam: 5, USA: 36, EU: 36, and UK: 29.")
                    context167 = ("For men's chino shorts - woven (regular waist band) in size 29, Waist: 29, Hips: 39, Inseam: 5, USA: 38, EU: 38, and UK: 30.")
                    context168 = ("For men's chino shorts - woven (regular waist band) in size 30, Waist: 30, Hips: 40, Inseam: 5, USA: 39, EU: 39, and UK: 30-31.")
                    context169 = ("For men's chino shorts - woven (regular waist band) in size 31, Waist: 31, Hips: 41, Inseam: 5, USA: 40, EU: 40, and UK: 31.")
                    context170 = ("For men's chino shorts - woven (regular waist band) in size 32, Waist: 32, Hips: 42, Inseam: 5, USA: 42, EU: 42, and UK: 32.")
                    context171 = ("For men's chino shorts - woven (regular waist band) in size 34, Waist: 34, Hips: 44, Inseam: 5, USA: 44, EU: 44, and UK: 34.")
                    context172 = ("For men's walk/board shorts - woven (garterized waist band) in size extra small, Waist: 28, Hips: 41, Inseam: 4.5, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context173 = ("For men's walk/board shorts - woven (garterized waist band) in size small, Waist: 30, Hips: 43, Inseam: 4.5, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context174 = ("For men's walk/board shorts - woven (garterized waist band) in size medium, Waist: 32, Hips: 45, Inseam: 4.5, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context175 = ("For men's walk/board shorts - woven (garterized waist band) in size large, Waist: 34, Hips: 47, Inseam: 4.5, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context176 = ("For men's walk/board shorts - woven (garterized waist band) in size extra large, Waist: 36, Hips: 49, Inseam: 4.5, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context177 = ("For men's boxer shorts (garterized waist band) in size double extra small, Waist: 20, Hips: 35, Inseam: 3.25, USA: 32, EU: 42, and UK: 8.")
                    context178 = ("For men's boxer shorts (garterized waist band) in size extra small, Waist: 22, Hips: 36, Inseam: 3.25, USA: 34, EU: 44, and UK: 10.")
                    context179 = ("For men's boxer shorts (garterized waist band) in size small, Waist: 24, Hips: 38, Inseam: 3.5, USA: 36, EU: 46, and UK: 12.")
                    context180 = ("For men's boxer shorts (garterized waist band) in size medium, Waist: 26, Hips: 40, Inseam: 3.5, USA: 38, EU: 48, and UK: 14.")
                    context181 = ("For men's boxer shorts (garterized waist band) in size large, Waist: 28, Hips: 42, Inseam: 3.75, USA: 40, EU: 50, and UK: 16.")
                    context182 = ("For men's boxer shorts (garterized waist band) in size extra large, Waist: 30, Hips: 44, Inseam: 3.75, USA: 42, EU: 52, and UK: 18.")
                    context183 = ("For men's boxer shorts (garterized waist band) in size double extra large, Waist: 32, Hips: 46, Inseam: 3.75, USA: 44, EU: 54, and UK: 20.")
                    context184 = ("For men's lounge - robe in size small, Waist: 44, Hips: 40, Inseam: 19.5, USA: 12-16, EU: 46-50, and UK: 36-40.")
                    context185 = ("For men's lounge - robe in size medium, Waist: 48, Hips: 42, Inseam: 19.5, USA: 18-20, EU: 52-54, and UK: 42-44.")
                    context186 = ("For men's lounge set - top slim fit in size double extra small, Waist: 34, Hips: 25.5, Inseam: 6.75, USA: 10, EU: 44, and UK: 34.")
                    context187 = ("For men's lounge set - top slim fit in size extra small, Waist: 36, Hips: 26, Inseam: 7, USA: 12, EU: 46, and UK: 36.")
                    context188 = ("For men's lounge set - top slim fit in size small, Waist: 38, Hips: 26.5, Inseam: 7.25, USA: 14, EU: 48, and UK: 38.")
                    context189 = ("For men's lounge set - top slim fit in size medium, Waist: 40, Hips: 27, Inseam: 7.5, USA: 16, EU: 50, and UK: 40.")
                    context190 = ("For men's lounge set - top slim fit in size large, Waist: 42, Hips: 27.5, Inseam: 7.75, USA: 18, EU: 52, and UK: 42.")
                    context191 = ("For men's lounge set - top slim fit in size extra large, Waist: 44, Hips: 28, Inseam: 8, USA: 20, EU: 54, and UK: 44.")
                    context192 = ("For men's lounge set - top slim fit in size double extra large, Waist: 46, Hips: 28.5, Inseam: 8.25, USA: 22, EU: 56, and UK: 46.")
                    context193 = ("For men's lounge set - bottom boxer shorts in size double extra small, Waist: 20, Hips: 36, Inseam: 3.25, USA: 33-35, EU: 33-35, and UK: 27-28.")
                    context194 = ("For men's lounge set - bottom boxer shorts in size extra small, Waist: 22, Hips: 38, Inseam: 3.25, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context195 = ("For men's lounge set - bottom boxer shorts in size small, Waist: 24, Hips: 40, Inseam: 3.5, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context196 = ("For men's lounge set - bottom boxer shorts in size medium, Waist: 26, Hips: 42, Inseam: 3.5, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context197 = ("For men's lounge set - bottom boxer shorts in size large, Waist: 28, Hips: 44, Inseam: 3.75, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context198 = ("For men's lounge set - bottom boxer shorts in size extra large, Waist: 30, Hips: 46, Inseam: 3.75, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context199 = ("For men's lounge set - bottom boxer shorts in size double extra large, Waist: 32, Hips: 48, Inseam: 3.75, USA: 47-48, EU: 47-48, and UK: 37-38.")
                    context200 = ("For men's footwear sliders unisex in size small, Ladies' US Size: 5, Men's US Size: N/A, Footbed Length: 23.5cm.")
                    context201 = ("For men's footwear sliders unisex in size medium, Ladies' US Size: 6-7, Men's US Size: N/A, Footbed Length: 24.9cm.")
                    context202 = ("For men's footwear sliders unisex in size large, Ladies' US Size: 8-9, Men's US Size: 6, Footbed Length: 26.3cm.")
                    context203 = ("For men's footwear sliders unisex in size extra large, Ladies' US Size: N/A, Men's US Size: 7-8, Footbed Length: 27.5cm.")
                    context204 = ("For men's footwear sliders unisex in size double extra large, Ladies' US Size: N/A, Men's US Size: 9-10, Footbed Length: 28.9cm.")
                    context205 = ("For men's footwear single band sliders, (US: 6, EU: 39, CM: 26.1), (US: 7, EU: 40, CM: 26.8), (US: 8, EU: 41, CM: 27.5), (US: 9, EU: 42, CM: 28.2), (US: 10, EU: 43, CM: 28.9).")
                    context206 = ("For men's footwear velcro sliders, (US: 6, EU: 39, CM: 26.1), (US: 7, EU: 40, CM: 26.8), (US: 8, EU: 41, CM: 27.5), (US: 9, EU: 42, CM: 28.2), (US: 10, EU: 43, CM: 28.5).")
                    context207 = ("For men's regular flipflops, (US: 6, EU: 39, CM: 26.6), (US: 7, EU: 40, CM: 27.3), (US: 8, EU: 41, CM: 28), (US: 9, EU: 42, CM: 28.7), (US: 10, EU: 43, CM: 29.4), then the slippers thicknes is 15mm.")
                    context208 = ("For men's lace-up sneakers, (US: 6, EU: 39, CM: 26.5), (US: 7, EU: 40, CM: 27), (US: 8, EU: 41, CM: 27.5), (US: 9, EU: 42, CM: 28), (US: 10, EU: 43, CM: 28.5).")
                    context209 = ("For men's lace-up runner shoes, (US: 6, EU: 39, CM: 26.5), (US: 7, EU: 40, CM: 27), (US: 8, EU: 41, CM: 27.5), (US: 9, EU: 42, CM: 28), (US: 10, EU: 43, CM: 28.5).")
                    context210 = ("For ladies' top tee - regular fit in size extra small, Chest: 34, Body Length: 22.5, Sleeve Length: 7, USA: 2, EU: 30, and UK: 6.")
                    context211 = ("For ladies' top tee - regular fit in size small, Chest: 36, Body Length: 23, Sleeve Length: 7.25, USA: 4, EU: 32, and UK: 8.")
                    context212 = ("For ladies' top tee - regular fit in size medium, Chest: 38, Body Length: 23.5, Sleeve Length: 7.5, USA: 6, EU: 36, and UK: 10.")
                    context213 = ("For ladies' top tee - regular fit in size large, Chest: 40, Body Length: 24, Sleeve Length: 7.75, USA: 8, EU: 40, and UK: 12.")
                    context214 = ("For ladies' top tee - regular fit in size extra large, Chest: 42, Body Length: 24.5, Sleeve Length: 8, USA: 10, EU: 42, and UK: 42.")
                    context215 = ("For ladies' top tee - boxy fit in size extra small, Chest: 39, Body Length: 19, Sleeve Length: 7.75, USA: 2, EU: 30, and UK: 6.")
                    context216 = ("For ladies' top tee - boxy fit in size small, Chest: 41, Body Length: 19.5, Sleeve Length: 8, USA: 4, EU: 32, and UK: 8.")
                    context217 = ("For ladies' top tee - boxy fit in size medium, Chest: 43, Body Length: 20, Sleeve Length: 8.25, USA: 6, EU: 36, and UK: 10.")
                    context218 = ("For ladies' top tee - boxy fit in size large, Chest: 45, Body Length: 20.5, Sleeve Length: 8.5, USA: 8, EU: 40, and UK: 12.")
                    context219 = ("For ladies' top tee - boxy fit in size extra large, Chest: 47, Body Length: 21, Sleeve Length: 8.75, USA: 10, EU: 42, and UK: 14.")
                    context220 = ("For ladies' v-neck - slim fit in size extra small, Chest: 31, Body Length: 22.5, Sleeve Length: 7.25, USA: 2, EU: 30, and UK: 6.")
                    context221 = ("For ladies' v-neck - slim fit in size small, Chest: 33, Body Length: 23, Sleeve Length: 7.5, USA: 4, EU: 32, and UK: 8.")
                    context222 = ("For ladies' v-neck - slim fit in size medium, Chest: 35, Body Length: 23.5, Sleeve Length: 7.75, USA: 6, EU: 36, and UK: 10.")
                    context223 = ("For ladies' v-neck - slim fit in size large, Chest: 37, Body Length: 24, Sleeve Length: 8, USA: 8, EU: 40, and UK: 12.")
                    context224 = ("For ladies' v-neck - slim fit in size extra large, Chest: 39, Body Length: 24.5, Sleeve Length: 8.25, USA: 10, EU: 42, and UK: 14.")
                    context225 = ("For ladies' rib slim fit tee in size extra small, Chest: 28, Body Length: 22.5, Sleeve Length: 7.75, USA: 2, EU: 30, and UK: 6.")
                    context226 = ("For ladies' rib slim fit tee in size small, Chest: 30, Body Length: 23, Sleeve Length: 8, USA: 4, EU: 32, and UK: 8.")
                    context227 = ("For ladies' rib slim fit tee in size medium, Chest: 30, Body Length: 23, Sleeve Length: 8, USA: 4, EU: 32, and UK: 8.")
                    context228 = ("For ladies' rib slim fit tee in size large, Chest: 34, Body Length: 24, Sleeve Length: 8.5, USA: 8, EU: 40, and UK: 12.")
                    context229 = ("For ladies' rib slim fit tee in size extra large, Chest: 36, Body Length: 24.5, Sleeve Length: 8.75, USA: 10, EU: 42, and UK: 14.")
                    context230 = ("For ladies' slim fit bundle tees in size extra small, Chest: 30, Body Length: 23, Sleeve Length: 4.75, USA: 2, EU: 30, and UK: 6.")
                    context231 = ("For ladies' slim fit bundle tees in size small, Chest: 32, Body Length: 23.5, Sleeve Length: 5, USA: 4, EU: 32, and UK: 8.")
                    context232 = ("For ladies' slim fit bundle tees in size medium, Chest: 34, Body Length: 24, Sleeve Length: 5.25, USA: 6, EU: 36, and UK: 10.")
                    context233 = ("For ladies' slim fit bundle tees in size large, Chest: 36, Body Length: 24.5, Sleeve Length: 5.5, USA: 8, EU: 40, and UK: 12.")
                    context234 = ("For ladies' slim fit bundle tees in size extra large, Chest: 38, Body Length: 25, Sleeve Length: 5.75, USA: 10, EU: 42, and UK: 14.")
                    context235 = ("For ladies' top tee oversized fit in size extra small, Chest: 38, Body Length: 24, Sleeve Length: 7.75, USA: 2, EU: 30, and UK: 6.")
                    context236 = ("For ladies' top tee oversized fit in size small, Chest: 40, Body Length: 24.5 , Sleeve Length: 8, USA: 4, EU: 32, and UK: 8.")
                    context237 = ("For ladies' top tee oversized fit in size medium, Chest: 42, Body Length: 25 , Sleeve Length: 8.25, USA: 6, EU: 36, and UK: 10.")
                    context238 = ("For ladies' top tee oversized fit in size large, Chest: 44, Body Length: 25.5, Sleeve Length: 8.5, USA: 8, EU: 40, and UK: 12.")
                    context239 = ("For ladies' top tee oversized fit in size extra large, Chest: 46, Body Length: 26, Sleeve Length: 8.75, USA: 10, EU: 42, and UK: 14.")
                    context240 = ("For ladies' top shirt - short sleeves in size extra small, Chest: 40, Body Length: 20.5, Sleeve Length: 7.5, USA: 2, EU: 30, and UK: 6.")
                    context241 = ("For ladies' top shirt - short sleeves in size small, Chest: 42, Body Length: 21, Sleeve Length: 7.75, USA: 4, EU: 32, and UK: 8.")
                    context242 = ("For ladies' top shirt - short sleeves in size medium, Chest: 44, Body Length: 21.5, Sleeve Length: 8, USA: 6, EU: 36, and UK: 10.")
                    context243 = ("For ladies' top shirt - short sleeves in size large, Chest: 46, Body Length: 22, Sleeve Length: 8.25, USA: 8, EU: 40, and UK: 12.")
                    context244 = ("For ladies' top shirt - short sleeves in size extra large, Chest: 48, Body Length: 22.5, Sleeve Length: 8.5, USA: 10, EU: 42, and UK: 14.")
                    context245 = ("For ladies' top shirt - long sleeves in size extra small, Chest: 40, Body Length: 27.5, Sleeve Length: 20.5, USA: 2, EU: 30, and UK: 6.")
                    context246 = ("For ladies' top shirt - long sleeves in size small, Chest: 42, Body Length: 28, Sleeve Length: 20.75, USA: 4, EU: 32, and UK: 8.")
                    context247 = ("For ladies' top shirt - long sleeves in size medium, Chest: 44, Body Length: 28.5, Sleeve Length: 21, USA: 6, EU: 36, and UK: 10.")
                    context248 = ("For ladies' top shirt - long sleeves in size large, Chest: 46, Body Length: 29, Sleeve Length: 21.25, USA: 8, EU: 40, and UK: 12.")
                    context249 = ("For ladies' top shirt - long sleeves in size extra large, Chest: 48, Body Length: 29.5, Sleeve Length: 21.5, USA: 10, EU: 42, and UK: 14.")
                    context250 = ("For ladies' top - sports bra in size extra small, Chest: 28, Body Length: 9, Sleeve Length: N/A, USA: 2, EU: 30, and UK: 6.")
                    context251 = ("For ladies' top - sports bra in size small, Chest: 30, Body Length: 9.5, Sleeve Length: N/A, USA: 4, EU: 32, and UK: 8.")
                    context252 = ("For ladies' top - sports bra in size medium, Chest: 32, Body Length: 10, Sleeve Length: N/A, USA: 6, EU: 36, and UK: 10.")
                    context253 = ("For ladies' top - sports bra in size large, Chest: 34, Body Length: 10.5, Sleeve Length: N/A, USA: 8, EU: 40, and UK: 12.")
                    context254 = ("For ladies' top - sports bra in size extra large, Chest: 36, Body Length: 11, Sleeve Length: N/A, USA: 10, EU: 42, and UK: 14.")
                    context255 = ("For ladies' top - hoodie in size extra small, Chest: 44, Body Length: 24.5, Sleeve Length: 17.5, USA: 2, EU: 30, and UK: 6.")
                    context256 = ("For ladies' top - hoodie in size small, Chest: 46, Body Length: 25, Sleeve Length: 17.75, USA: 4, EU: 32, and UK: 8.")
                    context257 = ("For ladies' top - hoodie in size medium, Chest: 48, Body Length: 25.5, Sleeve Length: 18, USA: 6, EU: 36, and UK: 10.")
                    context258 = ("For ladies' top - hoodie in size large, Chest: 50, Body Length: 26, Sleeve Length: 18.25, USA: 8, EU: 40, and UK: 12.")
                    context259 = ("For ladies' top - hoodie in size extra large, Chest: 52, Body Length: 26.5, Sleeve Length: 18.5, USA: 10, EU: 42, and UK: 14.")
                    context260 = ("For ladies' top - crop pullover in size extra small, Chest: 36, Body Length: 17.5, Sleeve Length: 18.75, USA: 2, EU: 30, and UK: 6.")
                    context261 = ("For ladies' top - crop pullover in size small, Chest: 38, Body Length: 18, Sleeve Length: 19, USA: 4, EU: 32, and UK: 8.")
                    context262 = ("For ladies' top - crop pullover in size medium, Chest: 40, Body Length: 18.5, Sleeve Length: 19.25, USA: 6, EU: 36, and UK: 10.")
                    context263 = ("For ladies' top - crop pullover in size large, Chest: 42, Body Length: 19, Sleeve Length: 19.5, USA: 8, EU: 40, and UK: 12.")
                    context264 = ("For ladies' top - crop pullover in size extra large, Chest: 44, Body Length: 19.5, Sleeve Length: 19.75, USA: 10, EU: 42, and UK: 14.")
                    context265 = ("For ladies' top - regular pullover in size extra small, Chest: 38, Body Length: 25.5, Sleeve Length: 20.75, USA: 2, EU: 30, and UK: 6.")
                    context266 = ("For ladies' top - regular pullover in size small, Chest: 40, Body Length: 26, Sleeve Length: 21, USA: 4, EU: 32, and UK: 8.")
                    context267 = ("For ladies' top - regular pullover in size medium, Chest: 42, Body Length: 26.5, Sleeve Length: 21.25, USA: 6, EU: 36, and UK: 10.")
                    context268 = ("For ladies' top - regular pullover in size large, Chest: 44, Body Length: 27, Sleeve Length: 21.5, USA: 8, EU: 40, and UK: 12.")
                    context269 = ("For ladies' top - regular pullover in size extra large, Chest: 46, Body Length: 27.5, Sleeve Length: 21.75, USA: 10, EU: 42, and UK: 14.")
                    context270 = ("For ladies' dress - regular fit in size extra small, Chest: 33, Body Length: 32.5, Sleeve Length: 7, USA: 2, EU: 30, and UK: 6.")
                    context271 = ("For ladies' dress - regular fit in size small, Chest: 35, Body Length: 33, Sleeve Length: 7.25, USA: 4, EU: 32, and UK: 8.")
                    context272 = ("For ladies' dress - regular fit in size medium, Chest: 37, Body Length: 33.5, Sleeve Length: 7.5, USA: 6, EU: 36, and UK: 10.")
                    context273 = ("For ladies' dress - regular fit in size large, Chest: 39, Body Length: 34, Sleeve Length: 7.75, USA: 8, EU: 40, and UK: 12.")
                    context274 = ("For ladies' dress - regular fit in size extra large, Chest: 41, Body Length: 34.5, Sleeve Length: 8, USA: 10, EU: 42, and UK: 14.")
                    context275 = ("For ladies' dress - oversized fit in size extra small, Chest: 38, Body Length: 33, Sleeve Length: 8.5, USA: 2, EU: 30, and UK: 6.")
                    context276 = ("For ladies' dress - oversized fit in size small, Chest: 40, Body Length: 33.5, Sleeve Length: 8.75, USA: 4, EU: 32, and UK: 8.")
                    context277 = ("For ladies' dress - oversized fit in size medium, Chest: 42, Body Length: 34, Sleeve Length: 9, USA: 6, EU: 36, and UK: 10.")
                    context278 = ("For ladies' dress - oversized fit in size large, Chest: 44, Body Length: 34.5, Sleeve Length: 9.25, USA: 8, EU: 40, and UK: 12.")
                    context279 = ("For ladies' dress - oversized fit in size extra large, Chest: 46, Body Length: 35, Sleeve Length: 9.5, USA: 10, EU: 42, and UK: 14.")
                    context280 = ("For ladies' dress - long sleeved regular fit in size extra small, Chest: 38, Body Length: 32, Sleeve Length: 20, USA: 2, EU: 30, and UK: 6.")
                    context281 = ("For ladies' dress - long sleeved regular fit in size small, Chest: 40, Body Length: 32.5, Sleeve Length: 20.25, USA: 4, EU: 32, and UK: 8.")
                    context282 = ("For ladies' dress - long sleeved regular fit in size medium, Chest: 42, Body Length: 33, Sleeve Length: 20.5, USA: 6, EU: 36, and UK: 10.")
                    context283 = ("For ladies' dress - long sleeved regular fit in size large, Chest: 44, Body Length: 33.5, Sleeve Length: 20.75, USA: 8, EU: 40, and UK: 12.")
                    context284 = ("For ladies' dress - long sleeved regular fit in size extra large, Chest: 46, Body Length: 34, Sleeve Length: 21, USA: 10, EU: 42, and UK: 14.")
                    context285 = ("For ladies' dress - baby doll dress in size extra small, Chest: 33, Body Length: 31, Sleeve Length: 7.75, USA: 2, EU: 30, and UK: 6.")
                    context286 = ("For ladies' dress - baby doll dress in size small, Chest: 35, Body Length: 31.5, Sleeve Length: 8, USA: 4, EU: 32, and UK: 8.")
                    context287 = ("For ladies' dress - baby doll dress in size medium, Chest: 37, Body Length: 32, Sleeve Length: 8.25, USA: 6, EU: 36, and UK: 10.")
                    context288 = ("For ladies' dress - baby doll dress in size large, Chest: 39, Body Length: 32.5, Sleeve Length: 8.5, USA: 8, EU: 40, and UK: 12.")
                    context289 = ("For ladies' dress - baby doll dress in size extra large, Chest: 41, Body Length: 33, Sleeve Length: 8.75, USA: 10, EU: 42, and UK: 14.")
                    context290 = ("For ladies' bottoms - skinny jeans (numerical sizing) in size 26, Waist: 26, Hip: 32,  Inseam: 26, USA: 4, EU: 32, and UK: 8.")
                    context291 = ("For ladies' bottoms - skinny jeans (numerical sizing) in size 27, Waist: 27, Hip: 33,  Inseam: 26, USA: 6, EU: 36, and UK: 10.")
                    context292 = ("For ladies' bottoms - skinny jeans (numerical sizing) in size 28, Waist: 28, Hip: 34,  Inseam: 26, USA: 8, EU: 38, and UK: 12.")
                    context293 = ("For ladies' bottoms - skinny jeans (numerical sizing) in size 29, Waist: 29, Hip: 35,  Inseam: 26, USA: 10, EU: 40, and UK: 14.")
                    context294 = ("For ladies' bottoms - skinny jeans (numerical sizing) in size 30, Waist: 30, Hip: 36,  Inseam: 26, USA: 10, EU: 42, and UK: 16.")
                    context295 = ("For ladies' bottoms - skinny jeans (numerical sizing) in size 32, Waist: 32, Hip: 38,  Inseam: 26, USA: 12, EU: 40, and UK: 14.")
                    context296 = ("For ladies' bottoms - skinny jeans (alpha sizing) in size extra small, Waist: 24, Hip: 36,  Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context297 = ("For ladies' bottoms - skinny jeans (alpha sizing) in size small, Waist: 26, Hip: 38, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context298 = ("For ladies' bottoms - skinny jeans (alpha sizing) in size medium, Waist: 28, Hip: 40, Inseam: 27, USA: 8, EU: 38, and UK: 12.")
                    context299 = ("For ladies' bottoms - skinny jeans (alpha sizing) in size large, Waist: 30, Hip: 42, Inseam: 27, USA: 10, EU: 40, and UK: 14.")
                    context300 = ("For ladies' bottoms - skinny jeans (alpha sizing) in size extra large, Waist: 32, Hip: 44, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context301 = ("For ladies' bottoms - basic straight jeans (numerical sizing) in size 26, Waist: 26, Hip: 36, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context302 = ("For ladies' bottoms - basic straight jeans (numerical sizing) in size 27, Waist: 27, Hip: 37, Inseam: 27, USA: 6, EU: 36, and UK: 8.")
                    context303 = ("For ladies' bottoms - basic straight jeans (numerical sizing) in size 28, Waist: 28, Hip: 38, Inseam: 27, USA: 6, EU: 38, and UK: 10.")
                    context304 = ("For ladies' bottoms - basic straight jeans (numerical sizing) in size 29, Waist: 39, Hip: 39, Inseam: 27, USA: 10, EU: 38, and UK: 12.")
                    context305 = ("For ladies' bottoms - basic straight jeans (numerical sizing) in size 30, Waist: 30, Hip: 40, Inseam: 27, USA: 10, EU: 40, and UK: 12.")
                    context306 = ("For ladies' bottoms - basic straight jeans (numerical sizing) in size 32, Waist: 32, Hip: 42, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context307 = ("For ladies' bottoms - wide leg jeans (alpha sizing) in size extra small, Waist: 24, Hip: 34, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context308 = ("For ladies' bottoms - wide leg jeans (alpha sizing) in size small, Waist: 26, Hip: 36, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context309 = ("For ladies' bottoms - wide leg jeans (alpha sizing) in size medium, Waist: 28, Hip: 38, Inseam: 27, USA: 8, EU: 38, and UK: 12.")
                    context310 = ("For ladies' bottoms - wide leg jeans (alpha sizing) in size large, Waist: 30, Hip: 40, Inseam: 27, USA: 10, EU: 40, and UK: 14.")
                    context311 = ("For ladies' bottoms - wide leg jeans (alpha sizing) in size extra large, Waist: 32, Hip: 42, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context312 = ("For ladies' bottoms - wide leg jeans (numeiracal sizing) in size 26, Waist: 25, Hip: 36.5, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context313 = ("For ladies' bottoms - wide leg jeans (numeiracal sizing) in size 27, Waist: 26, Hip: 37.5, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context314 = ("For ladies' bottoms - wide leg jeans (numeiracal sizing) in size 28, Waist: 27, Hip: 38.5, Inseam: 27, USA: 6, EU: 38, and UK: 10.")
                    context315 = ("For ladies' bottoms - wide leg jeans (numeiracal sizing) in size 29, Waist: 28, Hip: 39.5, Inseam: 27, USA: 10, EU: 38, and UK: 12.")
                    context316 = ("For ladies' bottoms - wide leg jeans (numeiracal sizing) in size 30, Waist: 29, Hip: 40.5, Inseam: 27, USA: 10, EU: 40, and UK: 12.")
                    context317 = ("For ladies' bottoms - wide leg jeans (numeiracal sizing) in size 32, Waist: 31, Hip: 42.5, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context318 = ("For ladies' bottoms - shorts (alpha sizing) in size extra small, Waist: 26, Hip: 36, Inseam: 2.5, USA: 4, EU: 32, and UK: 8.")
                    context319 = ("For ladies' bottoms - shorts (alpha sizing) in size small, Waist: 28, Hip: 38, Inseam: 2.5, USA: 6, EU: 36, and UK: 10.")
                    context320 = ("For ladies' bottoms - shorts (alpha sizing) in size medium, Waist: 30, Hip: 40, Inseam: 2.5, USA: 8, EU: 38, and UK: 12.")
                    context321 = ("For ladies' bottoms - shorts (alpha sizing) in size large, Waist: 32, Hip: 42, Inseam: 2.5, USA: 10, EU: 40, and UK: 14.")
                    context322 = ("For ladies' bottoms - shorts (alpha sizing) in size extra large, Waist: 34, Hip: 44, Inseam: 2.5, USA: 12, EU: 42, and UK: 16.")
                    context323 = ("For ladies' bottoms - shorts (numerical sizing) in size 26, Waist: 26, Hip: 36, Inseam: 2.5, USA: 4, EU: 32, and UK: 8.")
                    context324 = ("For ladies' bottoms - shorts (numerical sizing) in size 27, Waist: 27, Hip: 37, Inseam: 2.5, USA: 6, EU: 36, and UK: 10.")
                    context325 = ("For ladies' bottoms - shorts (numerical sizing) in size 28, Waist: 28, Hip: 38, Inseam: 2.5, USA: 6, EU: 38, and UK: 10.")
                    context326 = ("For ladies' bottoms - shorts (numerical sizing) in size 29, Waist: 29, Hip: 39, Inseam: 2.5, USA: 10, EU: 38, and UK: 12.")
                    context327 = ("For ladies' bottoms - shorts (numerical sizing) in size 30, Waist: 30, Hip: 40, Inseam: 2.5, USA: 10, EU: 40, and UK: 12.")
                    context328 = ("For ladies' bottoms - shorts (numerical sizing) in size 32, Waist: 32, Hip: 42, Inseam: 2.5, USA: 12, EU: 42, and UK: 16.")
                    context329 = ("For ladies' bottoms - biker shorts (alpha sizing) in size extra small, Waist: 25, Hip: 28, Inseam: 10, USA: 4, EU: 32, and UK: 8.")
                    context330 = ("For ladies' bottoms - biker shorts (alpha sizing) in size small, Waist: 27, Hip: 30, Inseam: 10, USA: 6, EU: 36, and UK: 10.")
                    context331 = ("For ladies' bottoms - biker shorts (alpha sizing) in size medium, Waist: 29, Hip: 32, Inseam: 10, USA: 8, EU: 38, and UK: 12.")
                    context332 = ("For ladies' bottoms - biker shorts (alpha sizing) in size large, Waist: 31, Hip: 34, Inseam: 10, USA: 10, EU: 40, and UK: 14.")
                    context333 = ("For ladies' bottoms - biker shorts (alpha sizing) in size extra large, Waist: 33, Hip: 36, Inseam: 10, USA: 12, EU: 42, and UK: 16.")
                    context334 = ("For ladies' bottoms - sports leggings with side pockets (alpha sizing) in size extra small, Waist: 23, Hip: 25, Inseam: 26, USA: 4, EU: 32, and UK: 8.")
                    context335 = ("For ladies' bottoms - sports leggings with side pockets (alpha sizing) in size small, Waist: 25, Hip: 27, Inseam: 26, USA: 6, EU: 36, and UK: 10.")
                    context336 = ("For ladies' bottoms - sports leggings with side pockets (alpha sizing) in size medium, Waist: 27, Hip: 29, Inseam: 26, USA: 8, EU: 38, and UK: 12.")
                    context337 = ("For ladies' bottoms - sports leggings with side pockets (alpha sizing) in size large, Waist: 29, Hip: 31, Inseam: 26, USA: 10, EU: 40, and UK: 14.")
                    context338 = ("For ladies' bottoms - sports leggings with side pockets (alpha sizing) in size extra large, Waist: 31, Hip: 33, Inseam: 26, USA: 12, EU: 42, and UK: 16.")
                    context339 = ("For ladies' bottoms - sports leggings without pockets (alpha sizing) in size extra small, Waist: 25, Hip: 28, Inseam: 26.5, USA: 4, EU: 32, and UK: 8.")
                    context340 = ("For ladies' bottoms - sports leggings without pockets (alpha sizing) in size small, Waist: 25, Hip: 28, Inseam: 26.5, USA: 4, EU: 32, and UK: 8.")
                    context341 = ("For ladies' bottoms - sports leggings without pockets (alpha sizing) in size medium, Waist: 29, Hip: 32, Inseam: 26.5, USA: 8, EU: 38, and UK: 12.")
                    context342 = ("For ladies' bottoms - sports leggings without pockets (alpha sizing) in size large, Waist: 31, Hip: 34, Inseam: 26.5, USA: 10, EU: 40, and UK: 14.")
                    context343 = ("For ladies' bottoms - sports leggings without pockets (alpha sizing) in size extra large, Waist: 33, Hip: 36, Inseam: 26.5, USA: 12, EU: 42, and UK: 16.")
                    context344 = ("For ladies' bottoms - track pants (mid-waist) in size extra small, Waist: 23, Hip: 35, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context345 = ("For ladies' bottoms - track pants (mid-waist) in size small, Waist: 25, Hip: 37, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context346 = ("For ladies' bottoms - track pants (mid-waist) in size medium, Waist: 27, Hip: 39, Inseam: 27, USA: 6, EU: 38, and UK: 10.")
                    context347 = ("For ladies' bottoms - track pants (mid-waist) in size large, Waist: 29, Hip: 41, Inseam: 27, USA: 10, EU: 38, and UK: 12.")
                    context348 = ("For ladies' bottoms - track pants (mid-waist) in size extra large, Waist: 31, Hip: 43, Inseam: 27, USA: 10, EU: 40, and UK: 12.")
                    context349 = ("For ladies' bottoms - track pants (high-waist) in size extra small, Waist: 21, Hip: 35, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context350 = ("For ladies' bottoms - track pants (high-waist) in size small, Waist: 23, Hip: 37, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context351 = ("For ladies' bottoms - track pants (high-waist) in size medium, Waist: 25, Hip: 39, Inseam: 27, USA: 8, EU: 38, and UK: 12.")
                    context352 = ("For ladies' bottoms - track pants (high-waist) in size large, Waist: 27, Hip: 41, Inseam: 27, USA: 10, EU: 40, and UK: 14.")
                    context353 = ("For ladies' bottoms - track pants (high-waist) in size extra large, Waist: 29, Hip: 43, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context354 = ("For ladies' bottoms - track pants with cuff in size extra small, Waist: 23, Hip: 35, Inseam: 26, USA: 4, EU: 32, and UK: 8.")
                    context355 = ("For ladies' bottoms - track pants with cuff in size small, Waist: 25, Hip: 37, Inseam: 26, USA: 6, EU: 36, and UK: 10.")
                    context356 = ("For ladies' bottoms - track pants with cuff in size medium, Waist: 27, Hip: 39, Inseam: 26, USA: 6, EU: 38, and UK: 10.")
                    context357 = ("For ladies' bottoms - track pants with cuff in size large, Waist: 29, Hip: 41, Inseam: 26, USA: 10, EU: 38, and UK: 12.")
                    context358 = ("For ladies' bottoms - track pants with cuff in size extra large, Waist: 31, Hip: 43, Inseam: 26, USA: 10, EU: 40, and UK: 12.")
                    context359 = ("For ladies' bottoms - urban tapered trousers (alpha sizing) in size extra small, Waist: 24, Hip: 34, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context360 = ("For ladies' bottoms - urban tapered trousers (alpha sizing) in size small, Waist: 26, Hip: 36, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context361 = ("For ladies' bottoms - urban tapered trousers (alpha sizing) in size medium, Waist: 28, Hip: 38, Inseam: 27, USA: 8, EU: 38, and UK: 12.")
                    context362 = ("For ladies' bottoms - urban tapered trousers (alpha sizing) in size large, Waist: 30, Hip: 40, Inseam: 27, USA: 10, EU: 40, and UK: 14.")
                    context363 = ("For ladies' bottoms - urban tapered trousers (alpha sizing) in size extra large, Waist: 32, Hip: 42, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context364 = ("For ladies' bottoms - urban tapered trousers (numerical sizing) in size 26, Waist: 26, Hip: 36, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context365 = ("For ladies' bottoms - urban tapered trousers (numerical sizing) in size 27, Waist: 27, Hip: 37, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context366 = ("For ladies' bottoms - urban tapered trousers (numerical sizing) in size 28, Waist: 28, Hip: 38, Inseam: 27, USA: 6, EU: 38, and UK: 10.")
                    context367 = ("For ladies' bottoms - urban tapered trousers (numerical sizing) in size 29, Waist: 29, Hip: 39, Inseam: 27, USA: 10, EU: 38, and UK: 12.")
                    context368 = ("For ladies' bottoms - urban tapered trousers (numerical sizing) in size 30, Waist: 30, Hip: 40, Inseam: 27, USA: 10, EU: 40, and UK: 12.")
                    context369 = ("For ladies' bottoms - urban tapered trousers (numerical sizing) in size 32, Waist: 32, Hip: 42, Inseam: 27, USA: 12, EU: 42, and UK: 16..")
                    context370 = ("For ladies' bottoms - urban wide trousers (alpha sizing) in size extra small, Waist: 24, Hip: 34, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context371 = ("For ladies' bottoms - urban wide trousers (alpha sizing) in size small, Waist: 26, Hip: 36, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context372 = ("For ladies' bottoms - urban wide trousers (alpha sizing) in size medium, Waist: 28, Hip: 38, Inseam: 27, USA: 8, EU: 38, and UK: 12.")
                    context373 = ("For ladies' bottoms - urban wide trousers (alpha sizing) in size large, Waist: 30, Hip: 40, Inseam: 27, USA: 10, EU: 40, and UK: 14.")
                    context374 = ("For ladies' bottoms - urban wide trousers (alpha sizing) in size extra large, Waist: 32, Hip: 42, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context375 = ("For ladies' bottoms - urban wide trousers (numerical sizing) in size 26, Waist: 26, Hip: 36, Inseam: 27, USA: 4, EU: 32, and UK: 8.")
                    context376 = ("For ladies' bottoms - urban wide trousers (numerical sizing) in size 27, Waist: 27, Hip: 37, Inseam: 27, USA: 6, EU: 36, and UK: 10.")
                    context377 = ("For ladies' bottoms - urban wide trousers (numerical sizing) in size 28, Waist: 28, Hip: 38, Inseam: 27, USA: 6, EU: 38, and UK: 10.")
                    context378 = ("For ladies' bottoms - urban wide trousers (numerical sizing) in size 29, Waist: 29, Hip: 39, Inseam: 27, USA: 10, EU: 38, and UK: 12.")
                    context379 = ("For ladies' bottoms - urban wide trousers (numerical sizing) in size 30, Waist: 30, Hip: 40, Inseam: 27, USA: 10, EU: 40, and UK: 12.")
                    context380 = ("For ladies' bottoms - urban wide trousers (numerical sizing) in size 32, Waist: 32, Hip: 42, Inseam: 27, USA: 12, EU: 42, and UK: 16.")
                    context381 = ("For men's loungewear robe in size small, Chest: 44, Body Length: 40, Sleeve Length: 19.5, USA: 12-16, EU: 46-50, and UK: 36-40.")
                    context382 = ("For men's loungewear robe in size medium, Chest: 48, Body Length: 42, Sleeve Length: 19.5, USA: 18-20, EU: 52-54, and UK: 42-44.")
                    context383 = ("For ladies' lounge set - top (oversized tee) in size extra small, Chest: 39, Body Length: 19, Sleeve Length: 7.75, USA: 2, EU: 30, and UK: 6.")
                    context384 = ("For ladies' lounge set - top (oversized tee) in size small, Chest: 41, Body Length: 19.5, Sleeve Length: 8, USA: 4, EU: 32, and UK: 8.")
                    context385 = ("For ladies' lounge set - top (oversized tee) in size medium, Chest: 43, Body Length: 20, Sleeve Length: 8.25, USA: 6, EU: 36, and UK: 10.")
                    context386 = ("For ladies' lounge set - top (oversized tee) in size large, Chest: 45, Body Length: 20.5, Sleeve Length: 8.5, USA: 8, EU: 40, and UK: 12.")
                    context387 = ("For ladies' lounge set - top (oversized tee) in size extra large, Chest: 47, Body Length: 21, Sleeve Length: 8.75, USA: 10, EU: 42, and UK: 14.")
                    context388 = ("For ladies' lounge set - bottom (shorts) in size extra small, Chest: 26, Body Length: 18, Sleeve Length: 2, USA: 4, EU: 32, and UK: 8.")
                    context389 = ("For ladies' lounge set - bottom (shorts) in size small, Chest: 28, Body Length: 19, Sleeve Length: 2, USA: 6, EU: 36, and UK: 10.")
                    context390 = ("For ladies' lounge set - bottom (shorts) in size medium, Chest: 30, Body Length: 20, Sleeve Length: 2, USA: 8, EU: 38, and UK: 12.")
                    context391 = ("For ladies' lounge set - bottom (shorts) in size large, Chest: 32, Body Length: 21, Sleeve Length: 2, USA: 10, EU: 40, and UK: 14.")
                    context392 = ("For ladies' lounge set - bottom (shorts) in size extra large, Chest: 34, Body Length: 22, Sleeve Length: 2, USA: 12, EU: 42, and UK: 16.")
                    context393 = ("For ladies' lounge set - top (spaghetti) in size extra small, Chest: 31, Body Length: 16, Sleeve Length: N/A, USA: 2, EU: 30, and UK: 6.")
                    context394 = ("For ladies' lounge set - top (spaghetti) in size small, Chest: 33, Body Length: 16.5, Sleeve Length: N/A, USA: 4, EU: 32, and UK: 8.")
                    context395 = ("For ladies' lounge set - top (spaghetti) in size medium, Chest: 35, Body Length: 17, Sleeve Length: N/A, USA: 6, EU: 36, and UK: 10.")
                    context396 = ("For ladies' lounge set - top (spaghetti) in size large, Chest: 37, Body Length: 17.5, Sleeve Length: N/A, USA: 8, EU: 40, and UK: 12.")
                    context397 = ("For ladies' lounge set - top (spaghetti) in size extra large, Chest: 39, Body Length: 18, Sleeve Length: N/A, USA: 10, EU: 42, and UK: 14.")
                    context398 = ("For ladies' lounge set - bottom (shorts) in size extra small, Chest: 26, Body Length: 36, Sleeve Length: 2, USA: 4, EU: 32, and UK: 8.")
                    context399 = ("For ladies' lounge set - bottom (shorts) in size small, Chest: 28, Body Length: 38, Sleeve Length: 2, USA: 6, EU: 36, and UK: 10.")
                    context400 = ("For ladies' lounge set - bottom (shorts) in size medium, Chest: 30, Body Length: 40, Sleeve Length: 2, USA: 8, EU: 38, and UK: 12.")
                    context401 = ("For ladies' lounge set - bottom (shorts) in size large, Chest: 32, Body Length: 42, Sleeve Length: 2, USA: 10, EU: 40, and UK: 14.")
                    context402 = ("For ladies' lounge set - bottom (shorts) in size extra large, Chest: 34, Body Length: 44, Sleeve Length: 2, USA: 12, EU: 42, and UK: 16.")
                    context403 = ("For ladies' footwear single band sliders, (US: 5, EU: 35, CM: 23.5), (US: 6, EU: 36, CM: 24.2), (US: 7, EU: 37, CM: 24.9), (US: 8, EU: 38, CM: 25.6), (US: 9, EU: 39, CM: 26.3).")
                    context404 = ("For ladies' footwear velcro sliders, (US: 5, EU: 35, CM: 23.5), (US: 6, EU: 36, CM: 24.2), (US: 7, EU: 37, CM: 24.9), (US: 8, EU: 38, CM: 25.6), (US: 9, EU: 39, CM: 26.3).")
                    context405 = ("For ladies' regular flipflops, (US: 5, EU: 35, CM: 23.2), (US: 6, EU: 36, CM: 23.9), (US: 7, EU: 37, CM: 24.6), (US: 8, EU: 38, CM: 25.3), (US: 9, EU: 39, CM: 26), then the slippers thicknes is 15mm.")
                    context406 = ("For ladies' lace-up sneakers, (US: 5, EU: 35, CM: 23.5), (US: 6, EU: 36, CM: 24), (US: 7, EU: 37, CM: 24.5), (US: 8, EU: 38, CM: 25), (US: 9, EU: 39, CM: 25.5).")
                    context407 = ("For ladies' runner shoes, (US: 5, EU: 35, CM: 23.5), (US: 6, EU: 36, CM: 24), (US: 7, EU: 37, CM: 24.5), (US: 8, EU: 38, CM: 25), (US: 9, EU: 39, CM: 25.5).")
                    context408 = ("For men's coed tee in size double extra small, Chest: 36, Body Length: 27.5, Sleeve Length: 7, USA: 10, EU: 44, and UK: 34.")
                    context409 = ("For men's coed tee in size extra small, Chest: 38, Body Length: 27.5, Sleeve Length: 7.25, USA: 12, EU: 46, and UK: 36.")
                    context410 = ("For men's coed tee in size small, Chest: 40, Body Length: 28, Sleeve Length: 7.5, USA: 14, EU: 48, and UK: 38.")
                    context411 = ("For men's coed tee in size medium, Chest: 42, Body Length: 28.5, Sleeve Length: 7.75, USA: 16, EU: 50, and UK: 40.")
                    context412 = ("For men's coed tee in size large, Chest: 44, Body Length: 29, Sleeve Length: 8, USA: 18, EU: 52, and UK: 42.")
                    context413 = ("For men's coed tee in size extra large, Chest: 46, Body Length: 30, Sleeve Length: 8.25, USA: 20, EU: 54, and UK: 44.")
                    context414 = ("For men's coed tee in size double extra large, Chest: 48, Body Length: 30.5, Sleeve Length: 8.5, USA: 22, EU: 56, and UK: 46.")
                    context415 = ("For men's coed tee in size triple extra large, Chest: 50, Body Length: 31, Sleeve Length: 8.75, USA: 24, EU: 58, and UK: 48.")
                    context416 = ("For ladies' coed tee in size double extra small, Chest: 36, Body Length: 27.5, Sleeve Length: 7, USA: 10, EU: 44, and UK: 34.")
                    context417 = ("For ladies' coed tee in size medium, Chest: 38, Body Length: 27.5, Sleeve Length: 7.25, USA: 12, EU: 46, and UK: 36.")
                    context418 = ("For ladies' coed tee in size large, Chest: 40, Body Length: 28, Sleeve Length: 7.5, USA: 14, EU: 48, and UK: 38.")
                    context419 = ("For ladies' coed tee in size extra large, Chest: 42, Body Length: 28.5, Sleeve Length: 7.75, USA: 16, EU: 50, and UK: 40.")
                    context420 = ("For men's coed pullover in size double extra small, Chest: 38, Body Length: 24, Sleeve Length: 21.25, USA: 10, EU: 44, and UK: 34.")
                    context421 = ("For men's coed pullover in size extra small, Chest: 40, Body Length: 25, Sleeve Length: 21.5, USA: 12, EU: 46, and UK: 36.")
                    context422 = ("For men's coed pullover in size small, Chest: 42, Body Length: 26, Sleeve Length: 21.75, USA: 14, EU: 48, and UK: 38.")
                    context423 = ("For men's coed pullover in size medium, Chest: 44, Body Length: 27, Sleeve Length: 22, USA: 16, EU: 50, and UK: 42.")
                    context424 = ("For men's coed pullover in size large, Chest: 46, Body Length: 28, Sleeve Length: 22.25, USA: 18, EU: 52, and UK: 42.")
                    context425 = ("For men's coed pullover in size extra large, Chest: 48, Body Length: 29, Sleeve Length: 22.5, USA: 20, EU: 54, and UK: 44.")
                    context426 = ("For men's coed pullover in size double extra large, Chest: 50, Body Length: 30, Sleeve Length: 22.75, USA: 22, EU: 56, and UK: 46.")
                    context427 = ("For men's coed pullover in size triple extra large, Chest: 52, Body Length: 31, Sleeve Length: 23, USA: 24, EU: 58, and UK: 48.")
                    context428 = ("For ladies' coed pullover in size small, Chest: 38, Body Length: 24, Sleeve Length: 21.25, USA: 10, EU: 44, and UK: 34.")
                    context429 = ("For ladies' coed pullover in size medium, Chest: 40, Body Length: 25, Sleeve Length: 21.5, USA: 12, EU: 46, and UK: 36.")
                    context430 = ("For ladies' coed pullover in size large, Chest: 42, Body Length: 26, Sleeve Length: 21.75, USA: 14, EU: 48, and UK: 38.")
                    context431 = ("or ladies' coed pullover in size extra large, Chest: 44, Body Length: 27, Sleeve Length: 22, USA: 16, EU: 50, and UK: 40.")
                    context432 = ("For men's coed woven trousers (urban straight trouser) in size double extra small, Waist: 22, Hip: 34, Inseam: 23, USA: 34-36, EU: 35-36, and UK: 27-28.")
                    context433 = ("For men's coed woven trousers (urban straight trouser) in size extra small, Waist: 24, Hip: 36, Inseam: 23, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context434 = ("For men's coed woven trousers (urban straight trouser) in size small, Waist: 26, Hip: 38, Inseam: 24, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context435 = ("For men's coed woven trousers (urban straight trouser) in size medium, Waist: 28, Hip: 40, Inseam: 24, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context436 = ("For men's coed woven trousers (urban straight trouser) in size large, Waist: 30, Hip: 42, Inseam: 25, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context437 = ("For men's coed woven trousers (urban straight trouser) in size extra large, Waist: 32, Hip: 44, Inseam: 25, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context438 = ("For ladies' coed woven trousers (urban straight trouser) in size small, Waist: 22, Hip: 34, Inseam: 23, USA: 34-36, EU: 35-36, and UK: 27-28.")
                    context439 = ("For ladies' coed woven trousers (urban straight trouser) in size medium, Waist: 24, Hip: 36, Inseam: 23, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context440 = ("For ladies' coed woven trousers (urban straight trouser) in size large, Waist: 26, Hip: 38, Inseam: 24, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context441 = ("For ladies' coed woven trousers (urban straight trouser) in size extra large, Waist: 28, Hip: 40, Inseam: 24, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context442 = ("For men's coed shorts (woven) in size double extra small, Waist: 27, Hip: 37, Inseam: 5, USA: 34-36, EU: 35-36, and UK: 27-28.")
                    context443 = ("For men's coed shorts (woven) in size extra small, Waist: 29, Hip: 39, Inseam: 5, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context444 = ("For men's coed shorts (woven) in size small, Waist: 31, Hip: 41, Inseam: 5, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context445 = ("For men's coed shorts (woven) in size medium, Waist: 33, Hip: 43, Inseam: 5, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context446 = ("For men's coed shorts (woven) in size large, Waist: 35, Hip: 45, Inseam: 5, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context447 = ("For men's coed shorts (woven) in size extra large, Waist: 37, Hip: 47, Inseam: 5, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context448 = ("For ladies' coed shorts (woven) in size small, Waist: 27, Hip: 37, Inseam: 5, USA: 34-36, EU: 35-36, and UK: 27-28.")
                    context449 = ("For ladies' coed shorts (woven) in size medium, Waist: 29, Hip: 39, Inseam: 5, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context450 = ("For ladies' coed shorts (woven) in size large, Waist: 31, Hip: 41, Inseam: 5, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context451 = ("For ladies' coed shorts (woven) in size extra large, Waist: 33, Hip: 43, Inseam: 5, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context452 = ("For men's coed woven trousers (tapered) in size double extra small, Waist: 22, Hip: 34, Inseam: 24, USA: 34-36, EU: 35-36, and UK: 27-28.")
                    context453 = ("For men's coed woven trousers (tapered) in size extra small, Waist: 24, Hip: 36, Inseam: 24, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context454 = ("For men's coed woven trousers (tapered) in size small, Waist: 26, Hip: 38, Inseam: 24, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context455 = ("For men's coed woven trousers (tapered) in size medium, Waist: 28, Hip: 40, Inseam: 24, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context456 = ("For men's coed woven trousers (tapered) in size large, Waist: 30, Hip: 42, Inseam: 24, USA: 43-44, EU: 43-44, and UK: 33-34.")
                    context457 = ("For men's coed woven trousers (tapered) in size extra large, Waist: 32, Hip: 44, Inseam: 24, USA: 45-46, EU: 45-46, and UK: 35-36.")
                    context458 = ("For ladies' coed woven trousers (tapered) in size small, Waist: 22, Hip: 34, Inseam: 24, USA: 34-36, EU: 35-36, and UK: 27-28.")
                    context459 = ("For ladies' coed woven trousers (tapered) in size medium, Waist: 24, Hip: 36, Inseam: 24, USA: 36-38, EU: 36-38, and UK: 29-30.")
                    context460 = ("For ladies' coed woven trousers (tapered) in size large, Waist: 26, Hip: 38, Inseam: 24, USA: 39-40, EU: 39-40, and UK: 30-31.")
                    context461 = ("For ladies' coed woven trousers (tapered) in size extra large, Waist: 28, Hip: 40, Inseam: 24, USA: 40-42, EU: 40-42, and UK: 31-32.")
                    context462 = ("For accessories - regular backpack, Length: 17 inches, Width: 11 inches, Depth: 5 inches, Strap Length: 16.5 inches, Strap Width: 2.75 inches.")
                    context463 = ("For accessories - large backpack, Length: 18 inches, Width: 12 inches, Depth: 6 inches, Strap Length: 16.5 inches, Strap Width: 2.75 inches.")
                    context464 = ("For accessories - drawstring backpack, Length: 17 inches and Width: 13 inches.")
                    context465 = ("For accessories - regular bum bag, Length: 4.5 inches, Width: 13.5 inches, Depth: 3 inches, Strap Length: 43 inches, Strap Width: 1.5 inches.")
                    context466 = ("For accessories - multi pockets bum bag, Length: 7.5 inches, Width: 13 inches, Depth: 3.5 inches.")
                    context467 = ("For accessories - oversized bum bag, Length: 9 inches, Width: 18 inches, Depth: 4 inches.")
                    context468 = ("For accessories - duffle bag, Length: 9.75 inches, Width: 18 inches, Depth: 9.75 inches.")
                    context469 = ("For accessories - mini sling bag, Length: 7 inches, Width: 5 inches, Depth: 1.5 inches, Strap Length: 53 inches, Strap Width: 1 inches.")
                    context470 = ("For accessories - regular sling bag, Length: 9 inches, Width: 7 inches, Depth: 2 inches, Strap Length: 53 inches, Strap Width: 1 inches.")
                    context471 = ("For accessories - mid sized messenger bag, Length: 10 inches, Width: 8 inches, Depth: 2.5 inches, Strap Length: 5.3 inches, Strap Width: 1.5 inches.")
                    context472 = ("For accessories - type 1 tote bag, Length: 16 inches, Width: 10 inches, Depth: 4 inches, Strap Length: 26 inches, Strap Width: 1.5 inches.")
                    context473 = ("For accessories - type 2 tote bag, Length: 16 inches, Width: 13.5 inches, Depth: N/A, Strap Length: 26 inches, Strap Width: 1 inches.")
                    context474 = ("For accessories - beanie, Circumference: 57 cm and Height: 8.89 cm.")
                    context475 = ("For accessories - bucket hat, Circumference: 59 cm, Height: 8.89 cm, Visor Length: 6.35 cm.")
                    context476 = ("For accessories - curved cap, Circumference: 57 cm, Height: 16.51 cm, Visor Length: 6.99 cm.")
                    context477 = ("For accessories - dad hat, Circumference: 57 cm, Height: 17.78 cm, Visor Length: 6.99 cm.")
                    context478 = ("For accessories - snap back, Circumference: 57 cm, Height: 16.51 cm, Visor Length: 6.35 cm.")
                    context479 = ("For accessories - trucker cap, Circumference: 57 cm, Height: 16.51 cm, Visor Length: 6.99 cm.")
                    context480 = ("For accessories - type 1 coin purse, Length: 4 inches, Width: 3 inches, Depth: 0.75 inch, Zipper Length: 3 3/4 inches.")
                    context481 = ("For accessories - type 2 coin purse, Length: 3 inches, Width: 4 inches, Depth: 0.75 inch, Zipper Length: 5 1/2 inches.")
                    context482 = ("For accessories - bifold wallet, Folded: (Length: 3.5 inches, Width: 3 1/2 inches, Depth: 0.5 inch), Bill Compartment: (Length: 4.5 inches and Height: 3 1/4 inches).")
                    context483 = ("For accessories - midsized bifold wallet, Folded: (Length: 3 1/4 inches, Width: 4 1/2 inches), Bill Compartment: (Length: 6 1/2 inches and Height: 3 3/4 inches).")
                    context484 = ("For accessories - trifold wallet, Folded: (Length: 4 1/4 inches, Width: 3 1/2 inches), Bill Compartment: (Length: 8 1/2 inches and Height: 3 1/4 inches).")
                    context485 = ("For accessories - zip around bifold wallet, Folded: (Length: 4 1/4 inches, Width: 3 1/2 inches, Depth: 0.5 inch), Bill Compartment: (Length: 8 1/2 inches and Height: 3 1/4 inches).")
                    context486 = ("For accessories - curved cap, Circumference: 57 cm, Height: 16.51 cm, Visor Length: 6.99 cm.")
                    context487 = ("For accessories - type 2 tote bag, Length: 16 inches, Width: 13.5 inches, Depth: N/A, Strap Length: 26 inches, Strap Width: 1 inches.")
                    context488 = ("For accessories - curved cap, Circumference: 57 cm, Height: 16.51 cm, Visor Length: 6.99 cm.")
                    context489 = ("For accessories - zip around bifold wallet, Folded: (Length: 4 1/4 inches, Width: 3 1/2 inches, Depth: 0.5 inch), Bill Compartment: (Length: 8 1/2 inches and Height: 3 1/4 inches).")
                    context490 = ("Yes, we offer international shipping. You can place your order online, and we'll ship it to your desired destination.")
                    context491 = ("Sale items are eligible for returns and exchanges within 14 days of purchase. Please ensure the item is in its original condition.")
                    context492 = ("You can stay updated on our upcoming sales and events by subscribing to our newsletter or following our social media channels.")
                    context493 = ("Yes, we offer personal shopping and styling services. Our team can help you find the perfect outfit tailored to your style.")
                    context494 = ("You can apply for our store credit card, which offers benefits like exclusive discounts, rewards, and special financing options.")
                    context495 = ("We welcome your feedback! You can provide feedback through our website, in-store, or by contacting our customer support.")
                    context496 = ("Yes, we can place a special order for items that are out of stock in your size. Please speak with a store associate for assistance.")
                    context497 = ("We offer gift wrapping services for a small fee. Your items will be beautifully packaged for a special touch.")
                    context498 = ("We release new collections regularly, typically with the changing seasons. Stay tuned for our latest arrivals.")
                    context499 = ("Yes, we offer the option to book private events or fashion shows in our store. Contact our events team for details and availability.")
                    context500 = ("Yes, we offer express and same-day delivery options for online orders in select locations.")
                    context501 = ("Absolutely! We have a loyalty program where you can earn rewards and exclusive discounts.")
                    context502 = ("We offer price adjustments for items purchased within seven days before a sale. Bring your receipt for verification.")
                    context503 = ("Yes, you can request clothing alterations at the time of purchase. Our team can assist you with adjustments.")
                    context504 = ("Yes, we offer special discounts for students and senior citizens. Please bring a valid ID for verification.")
                    context505 = ("We encourage clothing donations and recycling. You can inquire in-store about our donation programs.")
                    context506 = ("Certainly! We can assist with bulk orders for special events. Please contact our sales team for more details.")
                    context507 = ("Yes, we have a mobile app that makes online shopping convenient and offers exclusive deals.")
                    context508 = ("You can track your online order and check its status by logging into your account on our website or app.")
                    context509 = ("Yes, we offer gift-wrapping services for online orders, making it a perfect gift option.")
                    context510 = ("We offer store credit for returns within 30 days, or refunds in the original payment method.")
                    context511 = ("Yes, email subscribers receive exclusive discounts and early access to promotions.")
                    context512 = ("You can check the availability of items in a specific store location by contacting the store directly.")
                    context513 = ("We have robust security measures to protect customer data and adhere to strict privacy policies.")
                    context514 = ("Yes, we offer a gift registry service for weddings and special occasions. Visit our website for details.")
                    context515 = ("Yes, we have a referral program where you and your friends can enjoy discounts.")
                    context516 = ("You can apply for job openings by visiting our careers page on our website and submitting your application.")
                    context517 = ("Yes, we offer corporate gifting solutions, including branded and customized gift options.")
                    context518 = ("Yes, we can arrange to transfer items from other store locations for your pickup.")
                    context519 = ("You can schedule a fashion consultation or appointment by contacting our store and booking a time slot.")
                    context520 = ("Yes, we accept mobile payment options like Apple Pay and Google Pay for your convenience.")
                    context521 = ("Yes, you can exchange a gift for a different size or style within our return policy guidelines.")
                    context522 = ("Yes, we have a store newsletter that provides updates, promotions, and exclusive offers to subscribers.")
                    context523 = ("We have implemented measures to ensure a safe and comfortable shopping experience, including enhanced cleaning and safety protocols.")
                    context524 = ("For custom-made clothing, we require measurements and specific details to tailor the item to your preferences.")
                    context525 = ("Yes, you can use multiple payment methods for a single purchase. Our staff can assist with split payments.")
                    context526 = ("We will gladly replace or refund defective or damaged items. Please bring them to our attention within our return policy timeframe.")
                    context527 = ("We value your feedback! You can provide feedback in our stores or through our website.")
                    context528 = ("Yes, you can place orders for gifts and have them shipped to multiple addresses during the checkout process.")
                    context529 = ("You can sign up for a store credit card in-store or online, and eligibility requirements will be reviewed during the application process.")
                    context530 = ("Good morning! How can I assist you today?")
                    context531 = ("Hello! How can I help you?")
                    context532 = ("Good afternoon! What can I do for you?")
                    context533 = ("Good day! How may I assist you?")
                    context534 = ("Good evening! How can I be of service?")
                    context535 = ("Good night! How can I assist you at this hour?")
                    context536 = ("Good dawn! What brings you here so early?")
                    context537 = ("Greetings! How can I help you as the day winds down?")
                    context538 = ("Good morning and sunrise to you! How can I assist you?")
                    context539 = ("Good evening and sunset to you! What can I do for you?")
                    context540 = ("Hello! How may I assist you during this beautiful twilight?")
                    context541 = ("A tie bar, when worn properly, holds the tie in place and adds a stylish and functional element to a business outfit.")
                    context542 = ("When selecting dress shoes, consider classic styles like oxfords or derbies, and ensure they are well-polished and match the color of your belt.")
                    context543 = ("To incorporate seasonal trends, add accessories or select clothing items in seasonal colors while maintaining a professional core look.")
                    context544 = ("For a black-tie event, men should wear a black tuxedo, a black bow tie, and polished black dress shoes.")
                    context545 = ("Women have various options for footwear, including pumps, loafers, ankle boots, and closed-toe heels to match their business outfits.")
                    context546 = ("Statement jewelry can be worn in business attire but should be used sparingly to avoid appearing too flashy or distracting.")
                    context547 = ("A tailored dress shirt ensures a perfect fit and a sharp, professional appearance, as it's designed to complement your body shape.")
                    context548 = ("Transition from day to night by adding a blazer, statement accessories, and a change of shoes to elevate your look for evening events.")
                    context549 = ("Turtlenecks can be appropriate in business attire, especially in colder months, when worn with a well-fitted blazer or suit.")
                    context550 = ("Incorporate colors like navy or deep blue to convey confidence, or choose power colors like red or burgundy for a bolder statement.")
                    context551 = ("Good midnight! How can I assist you at this late hour?")
                    context552 = ("Good noon! How can I help you at this point in the day?")
                    context553 = ("Good morning! I'm here to help. What can I do for you?")
                    context554 = ("Hello! I'm here to assist you with anything you need.")
                    context555 = ("Good afternoon! Feel free to ask any questions or seek assistance.")
                    context556 = ("Hello! I'm here to assist you with anything you need. How can I help?")
                    context557 = ("Good evening! I'm here to help. How can I be of service?")
                    context558 = ("Good night! I'm here to assist you at this hour. What can I do for you?")
                    context559 = ("Good dawn! I'm here to assist you. How can I help at this early hour?")
                    context560 = ("Greetings! I'm here to help as the day winds down. What can I do for you?")
                    context561 = ("Good morning and sunrise to you! I'm here to help. How can I assist you today?")
                    context562 = ("Good evening and sunset to you! I'm here to assist you. What can I do for you?")
                    context563 = ("Hello! I'm here to help during this lovely twilight. How may I assist you?")
                    context564 = ("Good midnight! I'm here to assist you at this late hour. What can I do for you?")
                    context565 = ("Good noon! I'm here to assist you. What can I do for you at this time?")
                    context566 = ("Business casual attire typically includes slacks, a button-up shirt, and closed-toe shoes for men, and a blouse, slacks or skirt, and closed-toe shoes for women.")
                    context567 = ("Business formal attire is more formal than business casual and often includes a suit and tie for men, and a tailored suit or dress for women.")
                    context568 = ("Essential accessories may include a classic watch, a leather belt, and tasteful jewelry.")
                    context569 = ("To prolong the life of your suits, dry clean them only when necessary, hang them properly, and store them in a garment bag.")
                    context570 = ("Neutral colors like black, gray, navy, and white are often considered appropriate for business attire.")
                    context571 = ("A power tie, typically a bold and confident color, is worn to convey authority and confidence in business settings.")
                    context572 = ("Dress codes in the workplace help maintain a professional image and set a standard for appropriate attire.")
                    context573 = ("Choose lightweight, breathable fabrics like linen or cotton and opt for short-sleeved shirts or dresses.")
                    context574 = ("In many business settings, open-toed shoes are not considered appropriate; closed-toe shoes are a safer choice.")
                    context575 = ("Dress in professional attire that's slightly more formal than the company's dress code to make a positive impression.")
                    context576 = ("Denim is generally not considered business casual, but some workplaces allow it on specific days or with certain conditions.")
                    context577 = ("Wear solid, non-distracting colors, and make sure your top half looks professional as it's what's visible on camera.")
                    context578 = ("A blazer is a versatile piece that can elevate a business casual look and add a touch of formality.")
                    context579 = ("Sneakers are typically not considered appropriate in business settings; opt for more formal footwear.")
                    context580 = ("Personal grooming is crucial for making a positive impression and projecting professionalism.")
                    context581 = ("Look for sales, outlet stores, and second-hand shops to find high-quality attire at affordable prices.")
                    context582 = ("A well-fitted suit can enhance your confidence and convey attention to detail, which is highly regarded in the business world.")
                    context583 = ("Subtle patterns like stripes or checks are acceptable in business attire, but avoid loud or distracting prints.")
                    context584 = ("Women can accessorize with a stylish handbag, tasteful jewelry, and closed-toe heels or flats.")
                    context585 = ("Dress formally, often in a tailored suit for men and an elegant dress for women, to show respect for the occasion.")
                    context586 = ("Turtlenecks can be appropriate in a business casual setting, especially in colder months, as long as they are clean and well-maintained.")
                    context587 = ("A classic Windsor knot or a four-in-hand knot are suitable for a professional look in the business world.")
                    context588 = ("Incorporate your company's branding colors subtly in your attire, such as with a tie or scarf, to show alignment with the brand.")
                    context589 = ("Women can wear pantsuits in the business world, as they offer a professional and stylish alternative to dresses or skirts.")
                    context590 = ("Cufflinks add a touch of elegance and personal style to dress shirts with French cuffs, making them a symbol of sophistication.")
                    context591 = ("For a business-casual event, consider wearing chinos, a button-down shirt, and loafers for a polished yet relaxed appearance.")
                    context592 = ("All-black attire can be appropriate in a business setting, but it's important to add some variety with accessories or subtle patterns.")
                    context593 = ("Dress in professional attire that's slightly more formal than the company's dress code to make a positive impression.")
                    context594 = ("Look for sales, outlet stores, and second-hand shops to find high-quality attire at affordable prices.")
                    context595 = ("Business casual attire typically includes slacks, a button-up shirt, and closed-toe shoes for men, and a blouse, slacks or skirt, and closed-toe shoes for women.")
                    context596 = ("For a business dress, opt for fabrics like wool, wool blends, or high-quality synthetic materials for a professional look.")
                    context597 = ("A business skirt should typically fall just above or below the knee, ensuring a professional and modest appearance.")
                    context598 = ("Vests or waistcoats can add a touch of sophistication to business attire and are still considered stylish.")
                    context599 = ("A pocket square adds a touch of sophistication and personal style to a suit jacket, enhancing its overall appearance.")
                    context600 = ("Different collar styles suit different face shapes; for example, spread collars can complement round faces, while point collars work well with oval faces.")
                    context601 = ("When selecting a clothing store, consider factors such as location, product range, price range, and customer reviews to find the best fit for your shopping preferences.")
                    context602 = ("To find the best deals, sign up for newsletters, follow the store's social media accounts, and keep an eye on sales and clearance sections.")
                    context603 = ("Popular clothing store chains known for quality and selection include OXGN, Zara, H&M, Nordstrom, and Macy's.")
                    context604 = ("A well-organized store layout makes shopping easier, helping customers find what they need quickly and enhancing their overall shopping experience.")
                    context605 = ("Use the store's sizing charts, and don't hesitate to ask for assistance from store staff to ensure you find the right size.")
                    context606 = ("To make the changing room experience smoother, limit the number of items you bring in, and ensure you check the store's return policy.")
                    context607 = ("Stay updated on fashion trends by following fashion magazines, blogs, and influencers, and then shop for trendy items at your favorite clothing stores.")
                    context608 = ("Yes, there are eco-friendly clothing stores that offer sustainable and environmentally-conscious fashion choices, promoting ethical and responsible shopping.")
                    context609 = ("A personal shopper can assist customers in finding clothing that suits their style and needs, providing a tailored shopping experience.")
                    context610 = ("To make the most of sales and promotions, plan your shopping trips during sale events, and prioritize items you need.")
                    context611 = ("Shopping at thrift stores can be budget-friendly, so look for well-maintained and timeless pieces to score high-quality clothing at affordable prices.")
                    context612 = ("Yes, many clothing stores provide personalized tailoring services to ensure your clothing fits perfectly and is tailored to your preferences.")
                    context613 = ("Customer reviews and ratings provide valuable insights into the store's quality, service, and customer satisfaction, helping you make informed decisions.")
                    context614 = ("To maintain the quality of your clothing, follow care instructions, store garments properly, and handle stains promptly.")
                    context615 = ("Yes, some clothing stores specialize in specific fashion niches or styles, catering to unique tastes, such as vintage, bohemian, or streetwear.")
                    context616 = ("Loyalty programs can offer rewards, discounts, and exclusive deals for repeat customers, enhancing the shopping experience and building brand loyalty.")
                    context617 = ("To identify counterfeit items, research authentic designer features and logos, and be cautious when the price seems too good to be true.")
                    context618 = ("A fashion stylist can provide style advice, help you put together outfits, and suggest clothing items that complement your personal style.")
                    context619 = ("To make the most of the fitting room experience, take your time, try on different sizes, and assess the comfort and fit of each item.")
                    context620 = ("Several clothing stores, such as Patagonia, EILEEN FISHER, and Reformation, are known for their commitment to sustainable and ethical fashion practices.")
                    context621 = ("Budget your spending by setting a limit, making a shopping list, and considering cost per wear when evaluating potential purchases.")
                    context622 = ("Yes, many clothing stores collaborate with designers or brands to offer exclusive, limited-edition releases that attract fashion enthusiasts.")
                    context623 = ("When shopping online, consider checking size charts, reading product descriptions, and reviewing return policies to make informed decisions.")
                    context624 = ("Make sustainable choices by opting for eco-friendly materials, supporting ethical brands, and recycling or upcycling clothing items.")
                    context625 = ("When searching for a special occasion dress, consider the dress code, venue, and your personal style to find the perfect outfit.")
                    context626 = ("Avoid impulse buying by creating a shopping list, setting a budget, and taking your time to carefully consider each purchase.")
                    context627 = ("Many clothing stores now offer inclusive sizing options including us in OXGN store, catering to a wide range of body types and promoting body positivity.")
                    context628 = ("Seasonal sales and clearance events provide an opportunity to purchase clothing items at reduced prices, making it easier to update your wardrobe.")
                    context629 = ("To find clothing stores catering to specific age groups or demographics, read customer reviews, consult fashion magazines, and explore social media for recommendations.")
                    context630 = ("Clothing stores like Aerie and Target are known for their commitment to diversity and inclusivity, both in marketing and product offerings.")
                    context631 = ("Trying on clothing items allows you to assess the fit, comfort, and overall look of the garment, helping you make confident purchasing decisions.")
                    context632 = ("After a shopping spree, organize and declutter your closet by categorizing items, donating or selling clothing you no longer need, and maximizing storage space.")
                    context633 = ("When searching for the perfect pair of jeans, consider your body shape, preferred fit, and wash options, and consult store staff for assistance.")
                    context634 = ("Discover local boutiques by exploring your neighborhood, seeking recommendations from friends or online platforms, and attending local fashion events.")
                    context635 = ("Online customer reviews offer insights into a store's product quality, service, and customer experiences, aiding in your decision-making process.")
                    context636 = ("Brands and stores like Everlane, People Tree, and Outerknown prioritize fair labor practices and ethical sourcing in their fashion production.")
                    context637 = ("Look for clothing stores that offer customization or made-to-measure services to ensure your clothing fits perfectly and suits your unique style.")
                    context638 = ("Clothing store loyalty programs reward frequent shoppers with exclusive discounts, early access to sales, and other perks, enhancing their shopping experience.")
                    context639 = ("To find vintage or retro clothing stores, explore local thrift shops, consignment stores, or online marketplaces specializing in vintage fashion.")
                    context640 = ("Make environmentally conscious choices by opting for eco-friendly materials, supporting sustainable brands, and choosing clothing made with ethical practices.")
                    context641 = ("Inspect clothing items for sturdy stitching, quality fabrics, no defects, and attention to detail, which indicate good quality and durability.")
                    context642 = ("Department stores like Bloomingdale's and specialty stores like Men's Wearhouse offer a wide range of formal wear options for special events.")
                    context643 = ("To align your clothing purchases with your style and needs, plan your wardrobe, create a color palette, and focus on versatile pieces that can be mixed and matched.")
                    context644 = ("Brands like Aerie, Dove, and Fenty Beauty are recognized for promoting body positivity and diverse representation in their advertising campaigns.")
                    context645 = ("A fashion consultant can offer personalized style advice, recommend clothing items, and help customers discover fashion choices that suit their preferences.")
                    context646 = ("Look for timeless and versatile pieces like classic blazers, white shirts, and well-fitted jeans that can be worn in various outfits and styles.")
                    context647 = ("A fashion consultant can offer personalized style advice, recommend clothing items, and help customers discover fashion choices that suit their preferences.")
                    context648 = ("Calculate the cost per wear by dividing the item's price by the number of times you anticipate wearing it, helping you make cost-effective decisions.")
                    context649 = ("A clothing store's return and exchange policy ensures flexibility in case a purchase doesn't meet your expectations or fit as intended.")
                    context650 = ("Look for clothing stores that actively support charitable causes or donate a portion of their proceeds to social initiatives, making a positive impact through fashion.")
                    context651 = ("Customers often engage with clothing brands through social media, customer reviews, and feedback forms to share their opinions and preferences.")
                    context652 = ("Customer feedback is used to identify areas for improvement, refine product designs, and create offerings that align better with customer preferences.")
                    context653 = ("Clothing stores employ strategies like consistent pricing, inventory availability across channels, and responsive customer service to create a seamless shopping experience.")
                    context654 = ("Businesses protect customer data by implementing robust cybersecurity measures, encryption, and complying with data protection regulations to ensure security and privacy.")
                    context655 = ("Social media platforms provide a channel for clothing brands to engage with customers, gather feedback, and address concerns in real-time.")
                    context656 = ("Clothing retailers can encourage feedback through surveys, incentives, and user-friendly feedback platforms, making it convenient for customers to share their insights.")
                    context657 = ("Building trust involves delivering on promises, transparency, and addressing issues promptly, ultimately fostering strong and credible customer relationships.")
                    context658 = ("Personalization in marketing campaigns tailors content to individual customer preferences, increasing engagement and conversion rates for clothing brands.")
                    context659 = ("Effective complaint management involves prompt acknowledgment, thorough investigation, resolution, and feedback, ensuring customer satisfaction.")
                    context660 = ("Strategies include loyalty programs, personalized discounts, and exceptional customer service to retain loyal customers and reduce churn.")
                    context661 = ("Data analytics help clothing retailers analyze customer behavior and preferences, enabling data-driven decision-making and personalized experiences.")
                    context662 = ("Businesses benefit from CRM solutions with improved customer insights, streamlined processes, and the ability to nurture leads and retain customers effectively.")
                    context663 = ("Personalization in email marketing campaigns increases engagement and conversion rates by delivering relevant content tailored to individual preferences.")
                    context664 = ("Clothing brands should regularly update CRM strategies by monitoring customer trends, preferences, and feedback to remain responsive to changing needs.")
                    context665 = ("Omni-channel customer service ensures consistency and convenience across all customer touchpoints, whether in-store, online, or via mobile.")
                    context666 = ("A successful CRM strategy includes customer data management, personalization, analytics, and a customer-centric culture within the organization.")
                    context667 = ("Businesses safeguard data security and privacy through robust cybersecurity measures, data encryption, and compliance with privacy regulations.")
                    context668 = ("CRM allows clothing retailers to segment their customer base based on demographics, purchase history, and behavior, making it easier to tailor marketing efforts.")
                    context669 = ("Employee training and engagement are critical for providing excellent customer service, as motivated and well-informed staff are more likely to satisfy customer needs.")
                    context670 = ("Encouraging feedback involves creating user-friendly feedback platforms, conducting surveys, and offering incentives to make it easy for customers to share their insights.")
                    context671 = ("Omni-channel customer service ensures that customers experience consistent and convenient interactions across all channels, fostering strong and loyal relationships.")
                    context672 = ("Best practices include prompt acknowledgment, thorough investigation, resolution, and seeking customer feedback to ensure complaints are resolved effectively.")
                    context673 = ("CRM systems enable businesses to identify and nurture leads by tracking interactions and automating lead scoring to prioritize potential customers.")
                    context674 = ("Strategies for retaining loyal customers include offering rewards, personalized discounts, and exceptional customer service to reduce churn.")
                    context675 = ("Data analytics in CRM helps businesses analyze customer behavior patterns, enabling them to make data-driven decisions and personalize interactions.")
                    context676 = ("Clothing retailers measure effectiveness through metrics such as customer satisfaction, retention rates, lead conversion, and return on investment (ROI).")
                    context677 = ("Automation simplifies routine tasks, enhances efficiency, and ensures timely follow-ups, improving the overall customer experience.")
                    context678 = ("Personalization involves segmenting customers, tailoring content, and delivering personalized offers, promotions, and recommendations.")
                    context679 = ("Customer feedback provides valuable insights into customer preferences and areas for improvement, helping shape CRM strategies.")
                    context680 = ("Strategies include offering loyalty programs, conducting surveys, and tracking online behavior to collect and update customer data.")
                    context681 = ("Clothing brands protect customer data through data encryption, strict access controls, and compliance with privacy regulations to ensure responsible usage.")
                    context682 = ("Key components include customer data management, personalization, analytics, and a customer-centric culture within the organization.")
                    context683 = ("Social media platforms allow clothing brands to engage with customers, gather feedback, and address concerns in real-time, fostering strong customer relationships.")
                    context684 = ("Loyalty programs and incentives reward customers for repeat purchases, encourage brand loyalty, and boost customer retention.")
                    context685 = ("CRM systems help businesses track sales leads throughout the sales funnel, monitor progress, and measure lead conversion effectively.")
                    context686 = ("Clothing retailers adapt by monitoring trends, offering new styles, and communicating with customers to understand their preferences.")
                    context687 = ("Customer service representatives play a key role in resolving issues, addressing inquiries, and ensuring customers have a positive experience with the brand.")
                    context688 = ("Strategies include follow-up emails, post-purchase surveys, and personalized recommendations to maintain engagement after a sale.")
                    context689 = ("Ensuring consistency involves synchronizing inventory, pricing, and providing uniform customer service across all channels.")
                    context690 = ("Factors include product quality, personalized experiences, rewards programs, and excellent customer service that inspire loyalty and advocacy.")
                    context691 = ("CRM systems track and analyze customer interactions, helping brands identify and address emerging concerns promptly.")
                    context692 = ("Community engagement events provide opportunities for customers to connect with the brand, fostering a sense of belonging and stronger relationships.")
                    context693 = ("Effective returns and exchanges involve clear policies, hassle-free processes, and a focus on customer satisfaction to maintain positive relationships.")
                    context694 = ("Personalized loyalty programs offer tailored rewards, discounts, and incentives that enhance customer retention and engagement.")
                    context695 = ("Clothing brands invest in employee training programs to ensure staff can provide exceptional service and build strong customer relationships.")
                    context696 = ("CRM systems help identify high-value customers, allowing brands to offer exclusive benefits and rewards to enhance customer loyalty.")
                    context697 = ("Data analytics help brands analyze sales trends, customer preferences, and fashion shifts, allowing them to adjust inventory accordingly.")
                    context698 = ("Strategies include transparent communication, eco-friendly product lines, and participation in social and environmental initiatives to engage customers in sustainability.")
                    context699 = ("Brands can implement regular data cleansing processes and encourage customers to update their information to maintain accurate CRM data.")
                    context700 = ("A feedback and improvement loop allows brands to continually gather insights, make necessary improvements, and demonstrate responsiveness to customer needs, strengthening relationships.")
                
                    context_group_1 = (
                        context1 + "\n\n" + context2 + "\n\n" + context3 + "\n\n" + context4 + "\n\n" + context5 + "\n\n" +
                        context6 + "\n\n" + context7 + "\n\n" + context8 + "\n\n" + context9 + "\n\n" + context10 + "\n\n" +    
                        context11 + "\n\n" + context12 + "\n\n" + context13 + "\n\n" + context14 + "\n\n" + context15 + "\n\n" +
                        context16 + "\n\n" + context17 + "\n\n" + context18 + "\n\n" + context19 + "\n\n" + context20 + "\n\n" +
                        context21 + "\n\n" + context22 + "\n\n" + context23 + "\n\n" + context24 + "\n\n" + context25 + "\n\n" +
                        context26 + "\n\n" + context27 + "\n\n" + context28 + "\n\n" + context29 + "\n\n" + context30 + "\n\n" +
                        context31 + "\n\n" + context32 + "\n\n" + context33 + "\n\n" + context34 + "\n\n" + context35 + "\n\n" +
                        context36 + "\n\n" + context37 + "\n\n" + context38 + "\n\n" + context39 + "\n\n" + context40 
                    )

                    context_group_2 = (
                        context41 + "\n\n" + context42 + "\n\n" + context43 + "\n\n" + context44 + "\n\n" + context45 + "\n\n" +
                        context46 + "\n\n" + context47 + "\n\n" + context48 + "\n\n" + context49 + "\n\n" + context50 + "\n\n" +
                        context51 + "\n\n" + context52 + "\n\n" + context53 + "\n\n" + context54 + "\n\n" + context55 + "\n\n" +
                        context56 + "\n\n" + context57 + "\n\n" + context58 + "\n\n" + context59 + "\n\n" + context60 + "\n\n" +
                        context61 + "\n\n" + context62 + "\n\n" + context63 + "\n\n" + context64 + "\n\n" + context65 + "\n\n" +
                        context66 + "\n\n" + context67 + "\n\n" + context68 + "\n\n" + context69 + "\n\n" + context70 + "\n\n" +
                        context71 + "\n\n" + context72 + "\n\n" + context73 + "\n\n" + context74 + "\n\n" + context75 + "\n\n" +
                        context76 + "\n\n" + context77 + "\n\n" + context78 + "\n\n" + context79 + "\n\n" + context80 
                    )

                    context_group_3 = (
                        context81 + "\n\n" + context82 + "\n\n" + context83 + "\n\n" + context84 + "\n\n" + context85 + "\n\n" +
                        context86 + "\n\n" + context87 + "\n\n" + context88 + "\n\n" + context89 + "\n\n" + context90 + "\n\n" +
                        context91 + "\n\n" + context92 + "\n\n" + context93 + "\n\n" + context94 + "\n\n" + context95 + "\n\n" +
                        context96 + "\n\n" + context97 + "\n\n" + context98 + "\n\n" + context99 + "\n\n" + context100 + "\n\n" +
                        context101 + "\n\n" + context102 + "\n\n" + context103 + "\n\n" + context104 + "\n\n" + context105 + "\n\n" +
                        context106 + "\n\n" + context107 + "\n\n" + context108 + "\n\n" + context109 + "\n\n" + context110 + "\n\n" +
                        context111 + "\n\n" + context112 + "\n\n" + context113 + "\n\n" + context114 + "\n\n" + context115 + "\n\n" +
                        context116 + "\n\n" + context117 + "\n\n" + context118 + "\n\n" + context119 + "\n\n" + context120 
                    )    

                    context_group_4 = (
                        context121 + "\n\n" + context122 + "\n\n" + context123 + "\n\n" + context124 + "\n\n" + context125 + "\n\n" +
                        context126 + "\n\n" + context127 + "\n\n" + context128 + "\n\n" + context129 + "\n\n" + context130 + "\n\n" +
                        context131 + "\n\n" + context132 + "\n\n" + context133 + "\n\n" + context134 + "\n\n" + context135 + "\n\n" +
                        context136 + "\n\n" + context137 + "\n\n" + context138 + "\n\n" + context139 + "\n\n" + context140 + "\n\n" +   
                        context141 + "\n\n" + context142 + "\n\n" + context143 + "\n\n" + context144 + "\n\n" + context145 + "\n\n" +
                        context146 + "\n\n" + context147 + "\n\n" + context148 + "\n\n" + context149 + "\n\n" + context150 + "\n\n" +
                        context151 + "\n\n" + context152 + "\n\n" + context153 + "\n\n" + context154 + "\n\n" + context155 + "\n\n" +
                        context156 + "\n\n" + context157 + "\n\n" + context158 + "\n\n" + context159 + "\n\n" + context160 
                    )

                    context_group_5 = (
                        context161 + "\n\n" + context162 + "\n\n" + context163 + "\n\n" + context164 + "\n\n" + context165 + "\n\n" +
                        context166 + "\n\n" + context167 + "\n\n" + context168 + "\n\n" + context169 + "\n\n" + context170 + "\n\n" +
                        context171 + "\n\n" + context172 + "\n\n" + context173 + "\n\n" + context174 + "\n\n" + context175 + "\n\n" +
                        context176 + "\n\n" + context177 + "\n\n" + context178 + "\n\n" + context179 + "\n\n" + context180 + "\n\n" +
                        context181 + "\n\n" + context182 + "\n\n" + context183 + "\n\n" + context184 + "\n\n" + context185 + "\n\n" +
                        context186 + "\n\n" + context187 + "\n\n" + context188 + "\n\n" + context189 + "\n\n" + context190 + "\n\n" +
                        context191 + "\n\n" + context192 + "\n\n" + context193 + "\n\n" + context194 + "\n\n" + context195 + "\n\n" +
                        context196 + "\n\n" + context197 + "\n\n" + context198 + "\n\n" + context199 + "\n\n" + context200
                    )

                    context_group_6 = (
                        context201 + "\n\n" + context202 + "\n\n" + context203 + "\n\n" + context204 + "\n\n" + context205 + "\n\n" +
                        context206 + "\n\n" + context207 + "\n\n" + context208 + "\n\n" + context209 + "\n\n" + context210 + "\n\n" +
                        context211 + "\n\n" + context212 + "\n\n" + context213 + "\n\n" + context214 + "\n\n" + context215 + "\n\n" +
                        context216 + "\n\n" + context217 + "\n\n" + context218 + "\n\n" + context219 + "\n\n" + context220 + "\n\n" +
                        context221 + "\n\n" + context222 + "\n\n" + context223 + "\n\n" + context224 + "\n\n" + context225 + "\n\n" +
                        context226 + "\n\n" + context227 + "\n\n" + context228 + "\n\n" + context229 + "\n\n" + context230 + "\n\n" +
                        context231 + "\n\n" + context232 + "\n\n" + context233 + "\n\n" + context234 + "\n\n" + context235 + "\n\n" +
                        context236 + "\n\n" + context237 + "\n\n" + context238 + "\n\n" + context239 + "\n\n" + context240 
                    )

                    context_group_7 = (
                        context241 + "\n\n" + context242 + "\n\n" + context243 + "\n\n" + context244 + "\n\n" + context245 + "\n\n" +
                        context246 + "\n\n" + context247 + "\n\n" + context248 + "\n\n" + context249 + "\n\n" + context250 + "\n\n" +
                        context251 + "\n\n" + context252 + "\n\n" + context253 + "\n\n" + context254 + "\n\n" + context255 + "\n\n" +
                        context256 + "\n\n" + context257 + "\n\n" + context258 + "\n\n" + context259 + "\n\n" + context260 + "\n\n" +
                        context261 + "\n\n" + context262 + "\n\n" + context263 + "\n\n" + context264 + "\n\n" + context265 + "\n\n" +
                        context266 + "\n\n" + context267 + "\n\n" + context268 + "\n\n" + context269 + "\n\n" + context270 + "\n\n" +
                        context271 + "\n\n" + context272 + "\n\n" + context273 + "\n\n" + context274 + "\n\n" + context275 + "\n\n" +
                        context276 + "\n\n" + context277 + "\n\n" + context278 + "\n\n" + context279 + "\n\n" + context280 
                    )

                    context_group_8 = (
                        context281 + "\n\n" + context282 + "\n\n" + context283 + "\n\n" + context284 + "\n\n" + context285 + "\n\n" +
                        context286 + "\n\n" + context287 + "\n\n" + context288 + "\n\n" + context289 + "\n\n" + context290 + "\n\n" +
                        context291 + "\n\n" + context292 + "\n\n" + context293 + "\n\n" + context294 + "\n\n" + context295 + "\n\n" +
                        context296 + "\n\n" + context297 + "\n\n" + context298 + "\n\n" + context299 + "\n\n" + context300 + "\n\n" +
                        context301 + "\n\n" + context302 + "\n\n" + context303 + "\n\n" + context304 + "\n\n" + context305 + "\n\n" +
                        context306 + "\n\n" + context307 + "\n\n" + context308 + "\n\n" + context309 + "\n\n" + context310 + "\n\n" +
                        context311 + "\n\n" + context312 + "\n\n" + context313 + "\n\n" + context314 + "\n\n" + context315 + "\n\n" +
                        context316 + "\n\n" + context317 + "\n\n" + context318 + "\n\n" + context319 + "\n\n" + context320
                    )

                    context_group_9 = (
                        context321 + "\n\n" + context322 + "\n\n" + context323 + "\n\n" + context324 + "\n\n" + context325 + "\n\n" +
                        context326 + "\n\n" + context327 + "\n\n" + context328 + "\n\n" + context329 + "\n\n" + context330 + "\n\n" +
                        context331 + "\n\n" + context332 + "\n\n" + context333 + "\n\n" + context334 + "\n\n" + context335 + "\n\n" +
                        context336 + "\n\n" + context337 + "\n\n" + context338 + "\n\n" + context339 + "\n\n" + context340 + "\n\n" +
                        context341 + "\n\n" + context342 + "\n\n" + context343 + "\n\n" + context344 + "\n\n" + context345 + "\n\n" +
                        context346 + "\n\n" + context347 + "\n\n" + context348 + "\n\n" + context349 + "\n\n" + context350 + "\n\n" +
                        context351 + "\n\n" + context352 + "\n\n" + context353 + "\n\n" + context354 + "\n\n" + context355 + "\n\n" +
                        context356 + "\n\n" + context357 + "\n\n" + context358 + "\n\n" + context359 + "\n\n" + context360 
                    )

                    context_group_10 = (
                        context361 + "\n\n" + context362 + "\n\n" + context363 + "\n\n" + context364 + "\n\n" + context365 + "\n\n" +
                        context366 + "\n\n" + context367 + "\n\n" + context368 + "\n\n" + context369 + "\n\n" + context370 + "\n\n" +
                        context371 + "\n\n" + context372 + "\n\n" + context373 + "\n\n" + context374 + "\n\n" + context375 + "\n\n" +
                        context376 + "\n\n" + context377 + "\n\n" + context378 + "\n\n" + context379 + "\n\n" + context380 + "\n\n" +
                        context381 + "\n\n" + context382 + "\n\n" + context383 + "\n\n" + context384 + "\n\n" + context385 + "\n\n" +
                        context386 + "\n\n" + context387 + "\n\n" + context388 + "\n\n" + context389 + "\n\n" + context390 + "\n\n" +
                        context391 + "\n\n" + context392 + "\n\n" + context393 + "\n\n" + context394 + "\n\n" + context395 + "\n\n" +
                        context396 + "\n\n" + context397 + "\n\n" + context398 + "\n\n" + context399 + "\n\n" + context400
                    )

                    context_group_11 = (
                        context401 + "\n\n" + context402 + "\n\n" + context403 + "\n\n" + context404 + "\n\n" + context405 + "\n\n" +
                        context406 + "\n\n" + context407 + "\n\n" + context408 + "\n\n" + context409 + "\n\n" + context410 + "\n\n" +
                        context411 + "\n\n" + context412 + "\n\n" + context413 + "\n\n" + context414 + "\n\n" + context415 + "\n\n" +
                        context416 + "\n\n" + context417 + "\n\n" + context418 + "\n\n" + context419 + "\n\n" + context420 + "\n\n" +
                        context421 + "\n\n" + context422 + "\n\n" + context423 + "\n\n" + context424 + "\n\n" + context425 + "\n\n" +
                        context426 + "\n\n" + context427 + "\n\n" + context428 + "\n\n" + context429 + "\n\n" + context430 + "\n\n" +
                        context431 + "\n\n" + context432 + "\n\n" + context433 + "\n\n" + context434 + "\n\n" + context435 + "\n\n" +
                        context436 + "\n\n" + context437 + "\n\n" + context438 + "\n\n" + context439 + "\n\n" + context440
                    )

                    context_group_12 = (
                        context441 + "\n\n" + context442 + "\n\n" + context443 + "\n\n" + context444 + "\n\n" + context445 + "\n\n" +
                        context446 + "\n\n" + context447 + "\n\n" + context448 + "\n\n" + context449 + "\n\n" + context450 + "\n\n" +
                        context451 + "\n\n" + context452 + "\n\n" + context453 + "\n\n" + context454 + "\n\n" + context455 + "\n\n" +
                        context456 + "\n\n" + context457 + "\n\n" + context458 + "\n\n" + context459 + "\n\n" + context460 + "\n\n" +
                        context461 + "\n\n" + context462 + "\n\n" + context463 + "\n\n" + context464 + "\n\n" + context465 + "\n\n" +
                        context466 + "\n\n" + context467 + "\n\n" + context468 + "\n\n" + context469 + "\n\n" + context470 + "\n\n" +
                        context471 + "\n\n" + context472 + "\n\n" + context473 + "\n\n" + context474 + "\n\n" + context475 + "\n\n" +
                        context476 + "\n\n" + context477 + "\n\n" + context478 + "\n\n" + context479 + "\n\n" + context480 
                    )

                    context_group_13 = (
                        context481 + "\n\n" + context482 + "\n\n" + context483 + "\n\n" + context484 + "\n\n" + context485 + "\n\n" +
                        context486 + "\n\n" + context487 + "\n\n" + context488 + "\n\n" + context489 + "\n\n" + context490 + "\n\n" +
                        context491 + "\n\n" + context492 + "\n\n" + context493 + "\n\n" + context494 + "\n\n" + context495 + "\n\n" +
                        context496 + "\n\n" + context497 + "\n\n" + context498 + "\n\n" + context499 + "\n\n" + context500 + "\n\n" +
                        context501 + "\n\n" + context502 + "\n\n" + context503 + "\n\n" + context504 + "\n\n" + context505 + "\n\n" +
                        context506 + "\n\n" + context507 + "\n\n" + context508 + "\n\n" + context509 + "\n\n" + context510 + "\n\n" +
                        context511 + "\n\n" + context512 + "\n\n" + context513 + "\n\n" + context514 + "\n\n" + context515 + "\n\n" +
                        context516 + "\n\n" + context517 + "\n\n" + context518 + "\n\n" + context519 + "\n\n" + context520
                    )

                    context_group_14 = (
                        context521 + "\n\n" + context522 + "\n\n" + context523 + "\n\n" + context524 + "\n\n" + context525 + "\n\n" +
                        context526 + "\n\n" + context527 + "\n\n" + context528 + "\n\n" + context529 + "\n\n" + context530 + "\n\n" +
                        context531 + "\n\n" + context532 + "\n\n" + context533 + "\n\n" + context534 + "\n\n" + context535 + "\n\n" +
                        context536 + "\n\n" + context537 + "\n\n" + context538 + "\n\n" + context539 + "\n\n" + context540 + "\n\n" +
                        context541 + "\n\n" + context542 + "\n\n" + context543 + "\n\n" + context544 + "\n\n" + context545 + "\n\n" +
                        context546 + "\n\n" + context547 + "\n\n" + context548 + "\n\n" + context549 + "\n\n" + context550 + "\n\n" +
                        context551 + "\n\n" + context552 + "\n\n" + context553 + "\n\n" + context554 + "\n\n" + context555 + "\n\n" +
                        context556 + "\n\n" + context557 + "\n\n" + context558 + "\n\n" + context559 + "\n\n" + context560
                    )

                    context_group_15 = (
                        context561 + "\n\n" + context562 + "\n\n" + context563 + "\n\n" + context564 + "\n\n" + context565 + "\n\n" +
                        context566 + "\n\n" + context567 + "\n\n" + context568 + "\n\n" + context569 + "\n\n" + context570 + "\n\n" +
                        context571 + "\n\n" + context572 + "\n\n" + context573 + "\n\n" + context574 + "\n\n" + context575 + "\n\n" +
                        context576 + "\n\n" + context577 + "\n\n" + context578 + "\n\n" + context579 + "\n\n" + context580 + "\n\n" +
                        context581 + "\n\n" + context582 + "\n\n" + context583 + "\n\n" + context584 + "\n\n" + context585 + "\n\n" +
                        context586 + "\n\n" + context587 + "\n\n" + context588 + "\n\n" + context589 + "\n\n" + context590 + "\n\n" +
                        context591 + "\n\n" + context592 + "\n\n" + context593 + "\n\n" + context594 + "\n\n" + context595 + "\n\n" +
                        context596 + "\n\n" + context597 + "\n\n" + context598 + "\n\n" + context599 + "\n\n" + context600
                    )

                    context_group_16 = (
                        context601 + "\n\n" + context602 + "\n\n" + context603 + "\n\n" + context604 + "\n\n" + context605 + "\n\n" +
                        context606 + "\n\n" + context607 + "\n\n" + context608 + "\n\n" + context609 + "\n\n" + context610 + "\n\n" +
                        context611 + "\n\n" + context612 + "\n\n" + context613 + "\n\n" + context614 + "\n\n" + context615 + "\n\n" +
                        context616 + "\n\n" + context617 + "\n\n" + context618 + "\n\n" + context619 + "\n\n" + context620 + "\n\n" +
                        context621 + "\n\n" + context622 + "\n\n" + context623 + "\n\n" + context624 + "\n\n" + context625 + "\n\n" +
                        context626 + "\n\n" + context627 + "\n\n" + context628 + "\n\n" + context629 + "\n\n" + context630 + "\n\n" +
                        context631 + "\n\n" + context632 + "\n\n" + context633 + "\n\n" + context634 + "\n\n" + context635 + "\n\n" +
                        context636 + "\n\n" + context637 + "\n\n" + context638 + "\n\n" + context639 + "\n\n" + context640
                    )

                    context_group_17 = (
                        context641 + "\n\n" + context642 + "\n\n" + context643 + "\n\n" + context644 + "\n\n" + context645 + "\n\n" +
                        context646 + "\n\n" + context647 + "\n\n" + context648 + "\n\n" + context649 + "\n\n" + context650 + "\n\n" +
                        context651 + "\n\n" + context652 + "\n\n" + context653 + "\n\n" + context654 + "\n\n" + context655 + "\n\n" +
                        context656 + "\n\n" + context657 + "\n\n" + context658 + "\n\n" + context659 + "\n\n" + context660 + "\n\n" +
                        context661 + "\n\n" + context662 + "\n\n" + context663 + "\n\n" + context664 + "\n\n" + context665 + "\n\n" +
                        context666 + "\n\n" + context667 + "\n\n" + context668 + "\n\n" + context669 + "\n\n" + context670 + "\n\n" +
                        context671 + "\n\n" + context672 + "\n\n" + context673 + "\n\n" + context674 + "\n\n" + context675 + "\n\n" +
                        context676 + "\n\n" + context677 + "\n\n" + context678 + "\n\n" + context679 + "\n\n" + context680 + "\n\n" +
                        context681 + "\n\n" + context682 + "\n\n" + context683 + "\n\n" + context684 + "\n\n" + context685 + "\n\n" +
                        context686 + "\n\n" + context687 + "\n\n" + context688 + "\n\n" + context689 + "\n\n" + context690 + "\n\n" +
                        context691 + "\n\n" + context692 + "\n\n" + context693 + "\n\n" + context694 + "\n\n" + context695 + "\n\n" +
                        context696 + "\n\n" + context697 + "\n\n" + context698 + "\n\n" + context699 + "\n\n" + context700
                    )

                    all_contexts = context_group_1 + "\n\n" + context_group_2 + "\n\n" + context_group_3 + "\n\n" + context_group_4 + "\n\n" + context_group_5 + "\n\n" + context_group_6 + "\n\n" + context_group_7 + "\n\n" + context_group_8 + "\n\n" + context_group_9 + "\n\n" + context_group_10 + "\n\n" + context_group_11 + "\n\n" + context_group_12 + "\n\n" + context_group_13 + "\n\n" + context_group_14 + "\n\n" + context_group_15 + "\n\n" + context_group_16 + "\n\n" + context_group_17

                    true_labels = [1]  # Replace with the true labels
                    predicted_labels = [1]  # Replace with the predicted labels

                    # Calculate cosine similarity
                    all_contexts_list = all_contexts.split("\n")  # Split the string into a list of contexts
                    all_texts = [user_input] + all_contexts_list  # Concatenate user input and contexts

                    vectorizer = CountVectorizer().fit_transform(all_texts)
                    vectors = vectorizer.toarray()
                    cosine_similarities = cosine_similarity(vectors)

                    # Find the most similar context
                    most_similar_context_idx = cosine_similarities[0][1:].argmax()

                    # Add error handling for index out of range
                    if 0 <= most_similar_context_idx < len(all_contexts_list):
                        most_similar_context = all_contexts_list[most_similar_context_idx]

                    to_predict = []

                    # Create a context with the user's question
                    context = {
                        "context": all_contexts,
                        "qas": [{"question": user_input, "id": "user_question"}]
                    }

                    to_predict.append(context)

                    answers, _ = model.predict(to_predict)

                    # Extract the predicted answers and their confidence scores
                    predicted_answers = answers[0]["answer"]
                    confidences = answers[0].get("confidence", [1.0] * len(predicted_answers))

                    # Set a confidence threshold (adjust as needed)
                    confidence_threshold = 0.7

                    best_f1_score = 0
                    best_answer = None
                    em = 0  # Initialize EM to 0

                    # Record the time when the chatbot responds
                    chatbot_response_time = datetime.datetime.now()
                    formatted_chatbot_response_time = chatbot_response_time.strftime("%I:%M %p")

                    user_message = f"\nUser ({formatted_time}):\n{user_input}\n\n"
                    app.chatbot_proposed_convo.configure(state=customtkinter.NORMAL)
                    app.chatbot_proposed_convo.insert("end", user_message, "user")
                    app.chatbot_proposed_convo.tag_config("user", justify="right")

                    # Calculate metrics here
                    f1, em, precision = calculate_metrics(true_labels, predicted_labels)

                    if best_answer:
                        # Update the conversation textbox with user input, model response, and metrics
                        app.chatbot_proposed_convo.insert("end", f"DistilBERT+LSA Chatbot ({formatted_chatbot_response_time}):\n {best_answer}\n")
                    else:
                        # Update the conversation textbox with user input and model response
                        app.chatbot_proposed_convo.insert("end", f"DistilBERT+LSA Chatbot ({formatted_chatbot_response_time}):\n{most_similar_context}\n")

                    # Save the metrics to Excel 
                    save_proposed_metrics(f1, em, precision)
                    app.chatbot_frame_entry_proposed.delete(0, 'end')
                    app.update_idletasks()

                    # Record the end time
                    end_time = time.time()

                    # Calculate the execution time
                    execution_time = end_time - start_time

                    # Display the execution time
                    app.chatbot_proposed_convo.insert(customtkinter.END, f"Execution Time: {execution_time:.2f} seconds\n")
                    app.chatbot_proposed_convo.configure(state=customtkinter.DISABLED)
                    app.update_idletasks()
            finally:
                # Set the flag to indicate that processing is complete
                processing_in_progress = False

                # Stop and remove the progress bar when the process is complete
                progress_bar.stop()
                progress_bar.destroy()

        # Create a thread for the background task
        background_thread = threading.Thread(target=background_task)
        background_thread.start()

def open_baseline_metrics_excel(visualization_frame):
    file_path = r"C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Baseline Metrics Result\baseline_metrics.xlsx"
    if os.path.exists(file_path):
        os.startfile(file_path)
    else:
        messagebox.showinfo("File Not Found", "The Baseline Excel file does not exist.")

def open_proposed_metrics_excel(visualization_frame):
    file_path = r"C:\Users\Jude\Desktop\Thesis-1\LSADistilBERT\Proposed Metrics Result\proposed_metrics.xlsx"
    if os.path.exists(file_path):
        os.startfile(file_path)
    else:
        messagebox.showinfo("File Not Found", "The Proposed Excel file does not exist.")

if __name__ == "__main__":
    baseline_chatbot_response = Baseline_Chatbot()
    app = ModelSimulator()
    app.mainloop()