<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class OrderLine extends Model
{
    protected $fillable = ['order_id', 'product_sku', 'quantity', 'unit_price'];

    public function order()
    {
        return $this->belongsTo(Order::class);
    }
}
