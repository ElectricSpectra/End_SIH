# eoq_simple.py
"""
EOQ Calculator - FINAL CORRECT VERSION
Uses capacity utilization for realistic demand estimation
"""

import pandas as pd
from math import sqrt

# ===== PARAMETERS =====
ORDERING_COST = 10000
Z_SCORE = 2.05

# Project-based turnover (on CAPACITY, not current stock)
CAPACITY_UTILIZATION = 0.70  # Assume 70% average capacity utilization
TURNOVER_RATES = {
    'steel': 2.5,
    'conductor': 2.0,
    'concrete': 3.5,
    'insulator': 1.8,
    'hardware': 3.0
}

# Material-specific lead times
LEAD_TIME_DAYS = {
    'steel': 35,
    'conductor': 42,
    'concrete': 25,
    'insulator': 45,
    'hardware': 38
}

# Demand variability
DEMAND_CV = {
    'steel': 0.35,
    'conductor': 0.40,
    'concrete': 0.30,
    'insulator': 0.45,
    'hardware': 0.35
}

# Reorder threshold (% of capacity)
MIN_STOCK_THRESHOLD = 0.25  # Order when below 25% capacity


def check_inventory_status():
    """EOQ check using capacity-based demand"""
    
    warehouse_df = pd.read_csv('Datasets/warehouse.csv')
    results = []
    
    for _, wh in warehouse_df.iterrows():
        for material in ['steel', 'conductor', 'concrete', 'insulator', 'hardware']:
            
            current_stock = wh[f'{material}_current_stock_mt']
            capacity = wh[f'{material}_capacity_mt']
            holding_cost_annual = wh[f'storage_cost_{material}_per_mt_month'] * 12
            
            # ✅ CORRECT: Base demand on CAPACITY, not current stock
            turnover = TURNOVER_RATES[material]
            annual_demand = (capacity * CAPACITY_UTILIZATION) * turnover
            
            # Material-specific lead time
            lead_time = LEAD_TIME_DAYS[material]
            
            # Calculate EOQ
            eoq = sqrt((2 * annual_demand * ORDERING_COST) / holding_cost_annual)
            
            # Calculate Safety Stock
            daily_demand = annual_demand / 365
            demand_std = daily_demand * DEMAND_CV[material]
            safety_stock = Z_SCORE * demand_std * sqrt(lead_time)
            
            # Reorder Point
            reorder_point = (daily_demand * lead_time) + safety_stock
            
            # ✅ TWO CONDITIONS for ordering:
            # 1. Below reorder point (EOQ logic)
            # 2. Below minimum threshold (operational safety)
            stock_pct = current_stock / capacity
            below_threshold = stock_pct < MIN_STOCK_THRESHOLD
            below_reorder_point = current_stock <= reorder_point
            
            should_order = below_reorder_point or below_threshold
            
            # Days until stockout
            days_to_stockout = (current_stock - safety_stock) / daily_demand if daily_demand > 0 else 999
            
            # Priority level
            if stock_pct < 0.15 or days_to_stockout < 0:
                priority = "🔴 CRITICAL"
            elif stock_pct < 0.25 or days_to_stockout < 15:
                priority = "🟠 URGENT"
            elif stock_pct < 0.35 or days_to_stockout < 45:
                priority = "🟡 SOON"
            else:
                priority = "🟢 OK"
            
            # Order quantity: bring back to 70% capacity
            target_stock = capacity * CAPACITY_UTILIZATION
            order_qty = max(eoq, target_stock - current_stock) if should_order else 0
            
            results.append({
                'warehouse': wh['warehouse_name'],
                'region': wh['region'],
                'material': material.upper(),
                'current_stock': round(current_stock, 1),
                'capacity': round(capacity, 1),
                'stock_pct': round(stock_pct * 100, 1),
                'annual_demand_est': round(annual_demand, 1),
                'reorder_point': round(reorder_point, 1),
                'safety_stock': round(safety_stock, 1),
                'eoq': round(eoq, 1),
                'order_qty': round(order_qty, 1),
                'days_to_stockout': round(days_to_stockout, 1),
                'priority': priority,
                'reason': 'Below Threshold' if below_threshold else ('Below ROP' if below_reorder_point else 'OK'),
                'action': '📦 ORDER' if should_order else '✅ OK'
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    
    print("=" * 100)
    print("⚡ POWERGRID EOQ ANALYSIS - FINAL VERSION")
    print("=" * 100)
    print()
    
    df = check_inventory_status()
    
    # Filter items needing orders
    needs_order = df[df['action'] == '📦 ORDER']
    
    # Summary
    print(f"📊 INVENTORY STATUS:")
    print(f"   Total Items Analyzed: {len(df)}")
    print(f"   Need Reorder: {len(needs_order)} items ({len(needs_order)/len(df)*100:.1f}%)")
    print()
    
    if len(needs_order) > 0:
        
        # Show by priority
        for priority_level in ['🔴 CRITICAL', '🟠 URGENT', '🟡 SOON']:
            priority_items = needs_order[needs_order['priority'] == priority_level]
            if len(priority_items) > 0:
                print(f"\n{priority_level}: {len(priority_items)} items")
                print("-" * 100)
                print(priority_items[['warehouse', 'material', 'current_stock', 'capacity', 'stock_pct', 
                                      'reorder_point', 'order_qty', 'reason']].to_string(index=False))
        
        print()
        print("=" * 100)
        
        # Total order summary
        total_order_qty = needs_order['order_qty'].sum()
        print(f"📦 Total Order Quantity: {total_order_qty:,.0f} MT")
        print()
        
        # Regional summary
        print("📍 REORDERS BY REGION:")
        region_summary = needs_order.groupby('region').agg({
            'order_qty': 'sum',
            'warehouse': 'count'
        }).rename(columns={'warehouse': 'items_count'}).reset_index()
        print(region_summary.to_string(index=False))
        
    else:
        print("✅ ALL ITEMS HAVE ADEQUATE STOCK LEVELS")
    
    print()
    print("=" * 100)
    
    # Save report
    df.to_csv('eoq_final_report.csv', index=False)
    print("📄 Full report: eoq_final_report.csv")