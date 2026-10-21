<?php

namespace App\Http\Controllers;

use App\Models\Customer;
use Illuminate\Http\Request;

class CustomerController extends Controller
{
    public function index(Request $request)
    {
        $q = $request->query('q', '');
        return Customer::where('full_name', 'like', "%{$q}%")->limit(20)->get();
    }

    public function show(Customer $customer)
    {
        return $customer;
    }
}
