package com.example.cropdiseasetriage.screens;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.example.cropdiseasetriage.R;
import com.example.cropdiseasetriage.models.PredictionResult;
import java.util.List;

public class ResultActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        TextView tvTitle = findViewById(R.id.tvTitle);
        TextView tvStatusHeader = findViewById(R.id.tvStatusHeader);
        TextView tvDisease = findViewById(R.id.tvDisease);
        TextView tvConfidence = findViewById(R.id.tvConfidence);

        TextView lblDescription = findViewById(R.id.lblDescription);
        TextView tvDescription = findViewById(R.id.tvDescription);

        TextView lblSymptoms = findViewById(R.id.lblSymptoms);
        TextView tvSymptoms = findViewById(R.id.tvSymptoms);

        TextView lblCauses = findViewById(R.id.lblCauses);
        TextView tvCauses = findViewById(R.id.tvCauses);

        TextView lblPrevention = findViewById(R.id.lblPrevention);
        TextView tvPrevention = findViewById(R.id.tvPrevention);

        TextView lblTreatment = findViewById(R.id.lblTreatment);
        TextView tvTreatment = findViewById(R.id.tvTreatment);

        TextView lblTopPredictions = findViewById(R.id.lblTopPredictions);
        TextView tvTopPredictions = findViewById(R.id.tvTopPredictions);

        Button btnAnalyzeAnother = findViewById(R.id.btnAnalyzeAnother);

        PredictionResult result = (PredictionResult) getIntent().getSerializableExtra("result");

        if (result != null && result.prediction != null) {
            String plant = result.prediction.plant != null ? result.prediction.plant : "";
            String status = result.status != null ? result.status : "";

            if ("healthy".equalsIgnoreCase(status)) {
                tvTitle.setText(plant + " - Healthy");
                tvStatusHeader.setText("Status");
                tvDisease.setText("Healthy");

                // Hide disease-specific sections when healthy
                lblSymptoms.setVisibility(View.GONE);
                tvSymptoms.setVisibility(View.GONE);
                lblCauses.setVisibility(View.GONE);
                tvCauses.setVisibility(View.GONE);
                lblTopPredictions.setVisibility(View.GONE);
                tvTopPredictions.setVisibility(View.GONE);
            } else {
                String disease = result.prediction.disease != null ? result.prediction.disease : "Unknown Disease";
                tvTitle.setText(plant + " " + disease);
                tvStatusHeader.setText(R.string.likely_disease);
                tvDisease.setText(disease);
            }

            tvConfidence.setText(String.format("%.1f%%", result.prediction.confidence_percent));

            if (result.disease_info != null) {
                setTextOrHide(lblDescription, tvDescription, result.disease_info.description);
                setTextOrHideList(lblSymptoms, tvSymptoms, result.disease_info.symptoms);
                setTextOrHideList(lblCauses, tvCauses, result.disease_info.causes);
                setTextOrHideList(lblPrevention, tvPrevention, result.disease_info.prevention);
                setTextOrHideList(lblTreatment, tvTreatment, result.disease_info.recommended_action);
            }

            // Display top predictions if not healthy and data exists
            if (!"healthy".equalsIgnoreCase(status) && result.top_predictions != null && !result.top_predictions.isEmpty()) {
                StringBuilder topSb = new StringBuilder();
                for (PredictionResult.TopPrediction p : result.top_predictions) {
                    topSb.append("- ").append(p.disease)
                            .append(String.format(" (%.1f%%)", p.confidence_percent))
                            .append("\n");
                }
                tvTopPredictions.setText(topSb.toString().trim());
                lblTopPredictions.setVisibility(View.VISIBLE);
                tvTopPredictions.setVisibility(View.VISIBLE);
            } else if (!"healthy".equalsIgnoreCase(status)) {
                lblTopPredictions.setVisibility(View.GONE);
                tvTopPredictions.setVisibility(View.GONE);
            }
        }

        btnAnalyzeAnother.setOnClickListener(v -> {
            Intent intent = new Intent(this, HomeActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
            startActivity(intent);
            finish();
        });
    }

    private void setTextOrHide(TextView label, TextView content, String text) {
        if (text != null && !text.isEmpty()) {
            content.setText(text);
            label.setVisibility(View.VISIBLE);
            content.setVisibility(View.VISIBLE);
        } else {
            label.setVisibility(View.GONE);
            content.setVisibility(View.GONE);
        }
    }

    private void setTextOrHideList(TextView label, TextView content, List<String> list) {
        if (list != null && !list.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (String item : list) {
                sb.append("• ").append(item).append("\n");
            }
            content.setText(sb.toString().trim());
            label.setVisibility(View.VISIBLE);
            content.setVisibility(View.VISIBLE);
        } else {
            label.setVisibility(View.GONE);
            content.setVisibility(View.GONE);
        }
    }
}