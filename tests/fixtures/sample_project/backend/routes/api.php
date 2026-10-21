<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\OrderController;
use App\Http\Controllers\CustomerController;

Route::prefix('v1')->group(function () {
    Route::get('/health', fn () => ['ok' => true]);
});

Route::get('/orders', [OrderController::class, 'index']);
Route::get('/orders/{order}', [OrderController::class, 'show']);
Route::post('/orders/{order}/confirm', [OrderController::class, 'confirm']);
Route::apiResource('customers', CustomerController::class)->only(['index', 'show']);
