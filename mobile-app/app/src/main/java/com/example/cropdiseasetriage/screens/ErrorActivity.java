package com.example.cropdiseasetriage.screens;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;
import com.example.cropdiseasetriage.R;

public class ErrorActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_error);

        Button btnRetry = findViewById(R.id.btnRetry);
        Button btnHome = findViewById(R.id.btnHome);

        String imageUriString = getIntent().getStringExtra("imageUri");

        btnRetry.setOnClickListener(v -> {
            Intent intent = new Intent(this, ScanningActivity.class);
            if (imageUriString != null) {
                intent.putExtra("imageUri", imageUriString);
            }
            startActivity(intent);
            finish();
        });

        btnHome.setOnClickListener(v -> {
            Intent intent = new Intent(this, HomeActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
            startActivity(intent);
            finish();
        });
    }
}