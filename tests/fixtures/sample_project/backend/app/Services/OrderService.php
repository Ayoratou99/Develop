<?php

namespace App\Services;

use App\Models\Order;
use Illuminate\Support\Facades\Http;

class OrderService
{
    public function confirm(Order $order): Order
    {
        if ($order->status !== 'draft') {
            abort(422, 'Seule une commande brouillon peut être confirmée');
        }
        $order->status = 'confirmed';
        $order->confirmed_at = now();
        $order->save();
        Http::post('http://localhost:3001/notifications', [
            'user_id' => $order->customer_id,
            'message' => "Commande {$order->reference} confirmée",
        ]);
        return $order;
    }

    public function totalFor(Order $order): int
    {
        return $order->lines->sum(fn ($l) => $l->quantity * $l->unit_price);
    }
}
