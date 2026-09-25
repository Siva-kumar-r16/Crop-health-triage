package com.example.cropdiseasetriage.screens;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import com.example.cropdiseasetriage.R;
import java.io.File;
import java.io.IOException;

public class HomeActivity extends AppCompatActivity {

    private Uri cameraImageUri;

    private final ActivityResultLauncher<Intent> galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri selectedImageUri = result.getData().getData();
                    if (selectedImageUri != null) {
                        launchPreview(selectedImageUri);
                    }
                }
            }
    );

    private final ActivityResultLauncher<Uri> cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.TakePicture(),
            result -> {
                if (result) {
                    launchPreview(cameraImageUri);
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        Button btnTakePhoto = findViewById(R.id.btnTakePhoto);
        Button btnGallery = findViewById(R.id.btnGallery);

        btnTakePhoto.setOnClickListener(v -> openCamera());
        btnGallery.setOnClickListener(v -> openGallery());
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        galleryLauncher.launch(intent);
    }

    private void openCamera() {
        try {
            File photoFile = File.createTempFile("camera_photo", ".jpg", getCacheDir());
            cameraImageUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", photoFile);
            cameraLauncher.launch(cameraImageUri);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void launchPreview(Uri imageUri) {
        Intent intent = new Intent(this, PreviewActivity.class);
        intent.putExtra("imageUri", imageUri.toString());
        startActivity(intent);
    }
}