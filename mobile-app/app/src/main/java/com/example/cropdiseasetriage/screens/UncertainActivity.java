package com.example.cropdiseasetriage.screens;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.example.cropdiseasetriage.R;

public class UncertainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_uncertain);

        TextView tvMessage = findViewById(R.id.tvUncertainMessage);
        TextView tvSuggestion = findViewById(R.id.tvUncertainSuggestion);
        Button btnTakeAnother = findViewById(R.id.btnTakeAnother);

        String message = getIntent().getStringExtra("message");
        String suggestion = getIntent().getStringExtra("suggestion");

        if (message != null && !message.isEmpty()) {
            tvMessage.setText(message);
        }
        if (suggestion != null && !suggestion.isEmpty()) {
            tvSuggestion.setText(suggestion);
        }

        btnTakeAnother.setOnClickListener(v -> {
            Intent intent = new Intent(this, HomeActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
            startActivity(intent);
            finish();
        });
    }
}