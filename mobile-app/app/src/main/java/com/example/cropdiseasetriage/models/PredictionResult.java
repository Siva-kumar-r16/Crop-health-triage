package com.example.cropdiseasetriage.models;

import java.io.Serializable;
import java.util.List;

public class PredictionResult implements Serializable {
    public boolean success;
    public String status;
    public Prediction prediction;
    public DiseaseInfo disease_info;
    public List<TopPrediction> top_predictions;

    public static class Prediction implements Serializable {
        public int class_index;
        public String class_label;
        public String plant;
        public String disease;
        public double confidence;
        public double confidence_percent;
        public String confidence_level;
    }

    public static class DiseaseInfo implements Serializable {
        public String name;
        public String description;
        public List<String> symptoms;
        public List<String> causes;
        public List<String> prevention;
        public List<String> recommended_action;
    }

    public static class TopPrediction implements Serializable {
        public String class_label;
        public String plant;
        public String disease;
        public double confidence;
        public double confidence_percent;
    }
}