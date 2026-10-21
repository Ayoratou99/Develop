<?php

namespace App\Http\Controllers;

use App\Models\Order;
use App\Services\OrderService;
use Illuminate\Http\Request;

class OrderController extends Controller
{
    public function __construct(private OrderService $orders)
    {
    }

    public function index()
    {
        return Order::with('customer')->latest()->paginate(50);
    }

    public function show(Order $order)
    {
        return $order->load('customer', 'lines');
    }

    public function confirm(Request $request, Order $order)
    {
        return $this->orders->confirm($order);
    }
}
