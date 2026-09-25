package com.example.cropdiseasetriage.screens;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import com.example.cropdiseasetriage.R;

public class PreviewActivity extends AppCompatActivity {

    private String imageUriString;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_preview);

        ImageView imagePreview = findViewById(R.id.imagePreview);
        Button btnRetake = findViewById(R.id.btnRetake);
        Button btnAnalyze = findViewById(R.id.btnAnalyze);

        imageUriString = getIntent().getStringExtra("imageUri");
        if (imageUriString != null) {
            imagePreview.setImageURI(Uri.parse(imageUriString));
        }

        btnRetake.setOnClickListener(v -> finish());

        btnAnalyze.setOnClickListener(v -> {
            btnAnalyze.setEnabled(false);
            Intent intent = new Intent(this, ScanningActivity.class);
            intent.putExtra("imageUri", imageUriString);
            startActivity(intent);
            finish();
        });
    }
}