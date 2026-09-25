package com.example.cropdiseasetriage.services;

import com.example.cropdiseasetriage.models.PredictionResult;
import okhttp3.MultipartBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface ApiService {
    @Multipart
    @POST("predict")
    Call<PredictionResult> predict(
            @Part MultipartBody.Part file
    );
}