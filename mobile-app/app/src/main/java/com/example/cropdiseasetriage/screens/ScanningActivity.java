package com.example.cropdiseasetriage.screens;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import com.example.cropdiseasetriage.R;
import com.example.cropdiseasetriage.models.PredictionResult;
import com.example.cropdiseasetriage.services.ApiClient;
import com.example.cropdiseasetriage.services.ApiService;
import com.example.cropdiseasetriage.services.ImageUploadHelper;
import java.io.File;
import java.io.IOException;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ScanningActivity extends AppCompatActivity {

    private String imageUriString;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scanning);

        imageUriString = getIntent().getStringExtra("imageUri");

        if (imageUriString != null) {
            uploadImage();
        } else {
            showError();
        }
    }

    private void uploadImage() {
        try {
            File imageFile = ImageUploadHelper.getFileFromUri(this, Uri.parse(imageUriString));
            if (imageFile == null) {
                showError();
                return;
            }

            // Updated to use the "file" part name requested by the new FastAPI backend
            RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), imageFile);
            MultipartBody.Part body = MultipartBody.Part.createFormData("file", imageFile.getName(), requestFile);

            ApiService apiService = ApiClient.getClient().create(ApiService.class);
            Call<PredictionResult> call = apiService.predict(body);

            call.enqueue(new Callback<PredictionResult>() {
                @Override
                public void onResponse(Call<PredictionResult> call, Response<PredictionResult> response) {
                    if (response.isSuccessful() && response.body() != null) {
                        handleResponse(response.body());
                    } else {
                        showError();
                    }
                }

                @Override
                public void onFailure(Call<PredictionResult> call, Throwable t) {
                    showError();
                }
            });

        } catch (IOException e) {
            e.printStackTrace();
            showError();
        }
    }

    private void handleResponse(PredictionResult result) {
        if (!result.success) {
            showError();
            return;
        }

        if ("retake".equalsIgnoreCase(result.status)) {
            Intent intent = new Intent(this, UncertainActivity.class);
            intent.putExtra("message", "Image confidence is below 60%. Please take another clear photo of the leaf.");
            intent.putExtra("suggestion", "");
            startActivity(intent);
        } else {
            Intent intent = new Intent(this, ResultActivity.class);
            intent.putExtra("result", result);
            startActivity(intent);
        }
        finish();
    }

    private void showError() {
        Intent intent = new Intent(this, ErrorActivity.class);
        intent.putExtra("imageUri", imageUriString);
        startActivity(intent);
        finish();
    }
}