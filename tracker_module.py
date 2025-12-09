# tracker_module.py
"""
Resource Management Module for POWERGRID
Handles warehouses, vendors, and project tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()
# --- Email Configuration ---
ALERT_EMAIL = "sohansoma2806@gmail.com"

def send_low_stock_email(low_stock_items):
    """Send email alert for low stock items"""
    try:
        # Load email credentials from environment or Streamlit secrets
        sender_email = os.getenv("SENDER_EMAIL") or st.secrets.get("SENDER_EMAIL")
        email_password = os.getenv("EMAIL_PASSWORD") or st.secrets.get("EMAIL_PASSWORD")
        
        if not sender_email or not email_password:
            st.warning("⚠️ Email credentials not configured. Alert email not sent.")
            return False
        
        # Create email message
        em = EmailMessage()
        em['From'] = sender_email
        em['To'] = ALERT_EMAIL
        em['Subject'] = "🟡 URGENT: LOW STOCK INVENTORY ALERT - POWERGRID"
        
        # Build email body
        email_body = f"""
🚨 LOW STOCK ALERT - POWERGRID Warehouse Management System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Alert Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CRITICAL: {len(low_stock_items)} material(s) have fallen below 20% capacity threshold.
Immediate reordering recommended to prevent project delays.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOW STOCK DETAILS:
"""
        
        for idx, row in low_stock_items.iterrows():
            email_body += f"""
────────────────────────────────────────────────────────
Alert #{idx + 1}

🏭 Warehouse: {row['warehouse_name']}
📍 Location: {row['state']}, {row['region']}
🔧 Material: {row['material']}

📊 Stock Status:
   • Current Stock: {row['current_stock_mt']:.2f} MT
   • Total Capacity: {row['capacity_mt']:.2f} MT
   • Stock Level: {row['utilization_pct']:.2f}% (⚠️ BELOW 20%)
   • Available Space: {row['available_capacity_mt']:.2f} MT

💡 Recommended Action:
   • Reorder Quantity: {(row['capacity_mt'] * 0.5 - row['current_stock_mt']):.2f} MT
   • Target Stock Level: 50% capacity
"""
        
        email_body += f"""
────────────────────────────────────────────────────────

⚡ NEXT STEPS:
1. Review vendor availability for materials listed above
2. Place purchase orders to restore stock levels to 50% capacity
3. Coordinate with warehouse managers for delivery scheduling
4. Update stock levels in system after receipt confirmation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an automated alert from POWERGRID Resource Management System.
For questions, contact: warehouse.management@powergrid.in

© 2025 POWERGRID Corporation of India Limited
"""
        
        em.set_content(email_body)
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, email_password)
            smtp.sendmail(sender_email, ALERT_EMAIL, em.as_string())
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to send email alert: {str(e)}")
        return False

def send_stock_alert_email(alerts_df):
    """Send email alert for understocking and overstocking"""
    try:
        sender_email = os.getenv("SENDER_EMAIL") or st.secrets.get("SENDER_EMAIL")
        email_password = os.getenv("EMAIL_PASSWORD") or st.secrets.get("EMAIL_PASSWORD")
        
        if not sender_email or not email_password:
            st.warning("⚠️ Email credentials not configured. Alert email not sent.")
            return False
        
        understocking = alerts_df[alerts_df['alert_type'] == 'UNDERSTOCKING']
        overstocking = alerts_df[alerts_df['alert_type'] == 'OVERSTOCKING']
        
        em = EmailMessage()
        em['From'] = sender_email
        em['To'] = ALERT_EMAIL
        em['Subject'] = f"🚨 INVENTORY ALERT: {len(understocking)} Understocked | {len(overstocking)} Overstocked"
        
        email_body = f"""
🚨 INVENTORY ALERT - POWERGRID Warehouse Management System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Alert Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
• 🔴 UNDERSTOCKING: {len(understocking)} critical items (stock depleting faster than lead time)
• 🟠 OVERSTOCKING: {len(overstocking)} items (excess inventory, high storage costs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 UNDERSTOCKING ALERTS (URGENT - IMMEDIATE ACTION REQUIRED):
"""
        
        for idx, row in understocking.iterrows():
            email_body += f"""
────────────────────────────────────────────────────────
Priority: URGENT #{idx + 1}

🏭 Warehouse: {row['warehouse_name']}
📍 Location: {row['state']}, {row['region']}
🔧 Material: {row['material']}

📊 Current Status:
   • Current Stock: {row['current_stock_mt']:.2f} MT
   • Monthly Consumption: {row['monthly_consumption_mt']:.2f} MT/month
   • ⚠️ Stock Depletes In: {row['months_remaining']:.1f} MONTHS
   • Lead Time: {row['lead_time_months']} months
   
💡 URGENT ACTION:
   • 🚨 ORDER NOW: {row['reorder_qty_mt']:.2f} MT
   • Target: Restore to 4 months supply
   • {row['alert_message']}
"""
        
        if len(overstocking) > 0:
            email_body += f"""

🟠 OVERSTOCKING ALERTS (REVIEW - REDUCE STORAGE COSTS):
"""
            
            for idx, row in overstocking.iterrows():
                email_body += f"""
────────────────────────────────────────────────────────
Priority: REVIEW #{idx + 1}

🏭 Warehouse: {row['warehouse_name']}
📍 Location: {row['state']}, {row['region']}
🔧 Material: {row['material']}

📊 Current Status:
   • Current Stock: {row['current_stock_mt']:.2f} MT
   • Monthly Consumption: {row['monthly_consumption_mt']:.2f} MT/month
   • Stock Will Last: {row['months_remaining']:.1f} MONTHS
   
💡 RECOMMENDED ACTION:
   • Excess Stock: {row['reorder_qty_mt']:.2f} MT
   • {row['alert_message']}
   • Consider redistribution or reduced future orders
"""
        
        email_body += f"""
────────────────────────────────────────────────────────

⚡ NEXT STEPS:
1. URGENT: Process purchase orders for understocked items immediately
2. Review overstocked items for possible redistribution
3. Update procurement schedules to balance inventory levels
4. Coordinate with project managers on consumption forecasts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an automated alert from POWERGRID Resource Management System.
For questions, contact: warehouse.management@powergrid.in

© 2025 POWERGRID Corporation of India Limited
"""
        
        em.set_content(email_body)
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender_email, email_password)
            smtp.sendmail(sender_email, ALERT_EMAIL, em.as_string())
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to send email alert: {str(e)}")
        return False

# --- Data Loading Functions ---
@st.cache_data(ttl=60)
def load_projects_data():
    """Load projects data from CSV"""
    try:
        df = pd.read_csv('Datasets/powergrid_comprehensive_projects_bom_fixed_complete.csv')
        return df
    except Exception as e:
        st.error(f"Error loading projects data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_vendors_data():
    """Load vendors data from CSV"""
    try:
        df = pd.read_csv('Datasets/vendors.csv')
        return df
    except Exception as e:
        st.error(f"Error loading vendors data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_warehouse_data():
    """Load warehouse data from CSV"""
    try:
        df = pd.read_csv('Datasets/warehouse.csv')
        return df
    except Exception as e:
        st.error(f"Error loading warehouse data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_single_warehouse_data():
    """Load single warehouse detailed inventory from CSV"""
    try:
        df = pd.read_csv('Datasets/singlewarehouse.csv')
        return df
    except Exception as e:
        st.error(f"Error loading single warehouse data: {e}")
        return pd.DataFrame()

# --- Warehouse Safety Calculations ---
def calculate_warehouse_metrics(warehouse_df):
    """Calculate warehouse capacity utilization - FOCUS ON LOW STOCK ONLY"""
    
    metrics = []
    
    # Define LOW STOCK threshold only
    LOW_STOCK_THRESHOLD = 20  # Alert when stock falls below 20%
    
    for _, row in warehouse_df.iterrows():
        warehouse_id = row['warehouse_id']
        warehouse_name = row['warehouse_name']
        state = row['state']
        region = row['region']
        
        materials = ['steel', 'conductor', 'insulator', 'concrete', 'hardware']
        
        for material in materials:
            capacity_col = f'{material}_capacity_mt'
            stock_col = f'{material}_current_stock_mt'
            
            capacity = row[capacity_col]
            stock = row[stock_col]
            
            utilization_pct = (stock / capacity * 100) if capacity > 0 else 0
            available_capacity = capacity - stock
            
            # Determine status - ONLY LOW STOCK MATTERS
            if utilization_pct <= LOW_STOCK_THRESHOLD:
                status = "LOW_STOCK"
                status_color = "🟡"
            else:
                status = "NORMAL"
                status_color = "🟢"
            
            metrics.append({
                'warehouse_id': warehouse_id,
                'warehouse_name': warehouse_name,
                'state': state,
                'region': region,
                'material': material.title(),
                'capacity_mt': capacity,
                'current_stock_mt': stock,
                'available_capacity_mt': available_capacity,
                'utilization_pct': round(utilization_pct, 2),
                'status': status,
                'status_icon': status_color,
                'storage_cost_per_mt': row[f'storage_cost_{material}_per_mt_month']
            })
    
    return pd.DataFrame(metrics)

def calculate_consumption_and_alerts(warehouse_df, projects_df):
    """
    Calculate material consumption based on active projects and generate 
    understocking/overstocking alerts
    """
    alerts = []
    
    # Define thresholds
    LEAD_TIME_MONTHS = 2  # Average lead time for procurement (2 months)
    OVERSTOCKING_THRESHOLD_MONTHS = 6  # Stock lasting >6 months is overstocked
    SAFETY_BUFFER_MONTHS = 1  # Extra month buffer for safety
    
    # Filter active projects (Under Construction or Planned)
    active_projects = projects_df[projects_df['Status'].isin(['Under Construction', 'Planned'])]
    
    if len(active_projects) == 0:
        return pd.DataFrame(alerts)
    
    # Calculate total monthly consumption from active projects
    # Assume projects are uniformly distributed over their duration
    project_duration_months = 18  # Average project duration
    
    monthly_consumption = {
        'steel': active_projects['Total_Steel_MT'].sum() / project_duration_months,
        'conductor': active_projects['Total_Conductor_MT'].sum() / project_duration_months,
        'insulator': active_projects['Total_Insulators_Nos'].sum() / project_duration_months / 1000,  # Convert to MT equivalent
        'concrete': active_projects['Total_Concrete_Cum'].sum() / project_duration_months,
        'hardware': active_projects['Total_Hardware_MT'].sum() / project_duration_months if 'Total_Hardware_MT' in active_projects.columns else 0
    }
    
    # Analyze each warehouse
    for _, warehouse in warehouse_df.iterrows():
        warehouse_id = warehouse['warehouse_id']
        warehouse_name = warehouse['warehouse_name']
        state = warehouse['state']
        region = warehouse['region']
        
        # Check each material type
        for material, monthly_demand in monthly_consumption.items():
            capacity_col = f'{material}_capacity_mt'
            stock_col = f'{material}_current_stock_mt'
            
            if capacity_col not in warehouse.columns or stock_col not in warehouse.columns:
                continue
            
            capacity = warehouse[capacity_col]
            current_stock = warehouse[stock_col]
            
            if capacity == 0 or monthly_demand == 0:
                continue
            
            # Calculate metrics
            utilization_pct = (current_stock / capacity * 100)
            
            # Calculate months until stock depletes
            if monthly_demand > 0:
                months_remaining = current_stock / monthly_demand
            else:
                months_remaining = float('inf')
            
            # Determine alert type
            alert_type = None
            alert_icon = None
            alert_message = None
            priority = 0
            reorder_qty = 0
            
            # UNDERSTOCKING: Stock will run out before we can reorder
            if months_remaining < (LEAD_TIME_MONTHS + SAFETY_BUFFER_MONTHS):
                alert_type = "UNDERSTOCKING"
                alert_icon = "🔴"
                priority = 1  # Highest priority
                
                # Calculate how much to reorder to last for optimal period (4 months)
                target_months = 4
                reorder_qty = max(0, (monthly_demand * target_months) - current_stock)
                
                if months_remaining < LEAD_TIME_MONTHS:
                    alert_message = f"CRITICAL: Stock depletes in {months_remaining:.1f} months (less than lead time of {LEAD_TIME_MONTHS} months)"
                else:
                    alert_message = f"WARNING: Stock depletes in {months_remaining:.1f} months (minimal safety buffer)"
            
            # OVERSTOCKING: Stock will last too long (capital tied up, storage costs)
            elif months_remaining > OVERSTOCKING_THRESHOLD_MONTHS:
                alert_type = "OVERSTOCKING"
                alert_icon = "🟠"
                priority = 3  # Lower priority
                
                # Calculate excess stock
                optimal_months = 4
                optimal_stock = monthly_demand * optimal_months
                excess_qty = current_stock - optimal_stock
                
                alert_message = f"Excess stock: Will last {months_remaining:.1f} months (optimal: {optimal_months} months)"
                reorder_qty = -excess_qty  # Negative means excess
            
            # Only add alerts for actionable items
            if alert_type:
                alerts.append({
                    'warehouse_id': warehouse_id,
                    'warehouse_name': warehouse_name,
                    'state': state,
                    'region': region,
                    'material': material.title(),
                    'alert_type': alert_type,
                    'alert_icon': alert_icon,
                    'priority': priority,
                    'current_stock_mt': round(current_stock, 2),
                    'capacity_mt': round(capacity, 2),
                    'utilization_pct': round(utilization_pct, 2),
                    'monthly_consumption_mt': round(monthly_demand, 2),
                    'months_remaining': round(months_remaining, 2) if months_remaining != float('inf') else 999,
                    'lead_time_months': LEAD_TIME_MONTHS,
                    'reorder_qty_mt': round(abs(reorder_qty), 2),
                    'action_needed': reorder_qty > 0,  # True for understocking, False for overstocking
                    'alert_message': alert_message
                })
    
    alerts_df = pd.DataFrame(alerts)
    
    # Sort by priority (understocking first), then by months remaining
    if len(alerts_df) > 0:
        alerts_df = alerts_df.sort_values(['priority', 'months_remaining'], ascending=[True, True])
    
    return alerts_df

def calculate_single_warehouse_alerts(single_warehouse_df, projects_df):
    """
    Calculate understocking/overstocking alerts for single warehouse based on 
    project consumption and lead times
    """
    alerts = []
    
    if single_warehouse_df.empty or projects_df.empty:
        return pd.DataFrame(alerts)
    
    # Filter active projects
    active_projects = projects_df[projects_df['Status'].isin(['Under Construction', 'Planned'])]
    
    if len(active_projects) == 0:
        return pd.DataFrame(alerts)
    
    # Calculate monthly consumption (project duration = 18 months average)
    project_duration_months = 18
    
    # Map material categories to project columns
    consumption_mapping = {
        'Steel': active_projects['Total_Steel_MT'].sum() / project_duration_months,
        'Conductor': active_projects['Total_Conductor_MT'].sum() / project_duration_months,
        'Insulator': active_projects['Total_Insulators_Nos'].sum() / (project_duration_months * 1000),  # Convert to MT
        'Concrete': active_projects['Total_Concrete_Cum'].sum() / project_duration_months,
        'Hardware': 0  # Default if not available
    }
    
    # Check if Hardware column exists
    if 'Total_Hardware_MT' in active_projects.columns:
        consumption_mapping['Hardware'] = active_projects['Total_Hardware_MT'].sum() / project_duration_months
    
    # Analyze each material in warehouse
    for _, item in single_warehouse_df.iterrows():
        material_name = item['material_name']
        material_category = item['material_category']
        current_stock = item['current_stock_mt']
        capacity = item['capacity_mt']
        reorder_level = item['reorder_level_mt']
        safety_stock = item['safety_stock_mt']
        lead_time_days = item['lead_time_days']
        lead_time_months = lead_time_days / 30  # Convert to months
        
        # Get monthly consumption for this category
        monthly_consumption = consumption_mapping.get(material_category, 0)
        
        if monthly_consumption == 0:
            continue
        
        # Calculate metrics
        utilization_pct = (current_stock / capacity * 100) if capacity > 0 else 0
        months_remaining = current_stock / monthly_consumption if monthly_consumption > 0 else float('inf')
        
        # Thresholds
        SAFETY_BUFFER_MONTHS = 1
        OVERSTOCKING_THRESHOLD_MONTHS = 6
        
        alert_type = None
        alert_icon = None
        alert_message = None
        priority = 0
        action_qty = 0
        
        # CRITICAL: Below safety stock
        if current_stock < safety_stock:
            alert_type = "CRITICAL_UNDERSTOCK"
            alert_icon = "🔴"
            priority = 1
            target_stock = reorder_level + safety_stock
            action_qty = target_stock - current_stock
            alert_message = f"CRITICAL: Below safety stock! Current: {current_stock:.2f} MT, Safety: {safety_stock:.2f} MT"
        
        # UNDERSTOCKING: Will run out before lead time
        elif months_remaining < (lead_time_months + SAFETY_BUFFER_MONTHS):
            alert_type = "UNDERSTOCKING"
            alert_icon = "🟠"
            priority = 2
            target_stock = monthly_consumption * 4  # 4 months supply
            action_qty = max(0, target_stock - current_stock)
            
            if months_remaining < lead_time_months:
                alert_message = f"URGENT: Stock depletes in {months_remaining:.1f} months (Lead time: {lead_time_months:.1f} months)"
            else:
                alert_message = f"WARNING: Stock depletes in {months_remaining:.1f} months. Order before lead time expires."
        
        # OVERSTOCKING: Too much inventory
        elif months_remaining > OVERSTOCKING_THRESHOLD_MONTHS:
            alert_type = "OVERSTOCKING"
            alert_icon = "🟡"
            priority = 3
            action_qty = current_stock - (monthly_consumption * 4)  # Excess over 4 months supply
            alert_message = f"Excess stock: Will last {months_remaining:.1f} months (optimal: 4 months)"
        
        # NEAR REORDER LEVEL
        elif current_stock <= reorder_level:
            alert_type = "REORDER_NEEDED"
            alert_icon = "⚠️"
            priority = 2
            action_qty = (reorder_level + safety_stock) - current_stock
            alert_message = f"Stock at reorder level. Order: {action_qty:.2f} MT"
        
        # Only add actionable alerts
        if alert_type:
            alerts.append({
                'warehouse_name': item['warehouse_name'],
                'state': item['state'],
                'region': item['region'],
                'material_name': material_name,
                'material_category': material_category,
                'alert_type': alert_type,
                'alert_icon': alert_icon,
                'priority': priority,
                'current_stock_mt': round(current_stock, 2),
                'capacity_mt': round(capacity, 2),
                'reorder_level_mt': round(reorder_level, 2),
                'safety_stock_mt': round(safety_stock, 2),
                'utilization_pct': round(utilization_pct, 2),
                'monthly_consumption_mt': round(monthly_consumption, 2),
                'months_remaining': round(months_remaining, 2) if months_remaining != float('inf') else 999,
                'lead_time_days': lead_time_days,
                'lead_time_months': round(lead_time_months, 2),
                'action_qty_mt': round(abs(action_qty), 2),
                'supplier_name': item['supplier_name'],
                'storage_location': item['storage_location'],
                'last_updated': item['last_updated'],
                'alert_message': alert_message
            })
    
    alerts_df = pd.DataFrame(alerts)
    
    # Sort by priority and months remaining
    if len(alerts_df) > 0:
        alerts_df = alerts_df.sort_values(['priority', 'months_remaining'], ascending=[True, True])
    
    return alerts_df

def calculate_single_warehouse_alerts_from_procurement(single_warehouse_df, appv6_projects):
    """
    Calculate alerts using actual procurement data from appV6.py projects
    Enhanced with 6-level categorization and smart material filtering
    """
    alerts = []
    
    if single_warehouse_df.empty or not appv6_projects:
        return pd.DataFrame(alerts)
    
    # Track which materials are actually needed by projects
    materials_in_active_projects = set()
    
    # Calculate monthly consumption from actual procurement schedules
    monthly_consumption_by_category = {
        'Steel': 0,
        'Conductor': 0,
        'Insulator': 0,
        'Concrete': 0,
        'Hardware': 0
    }
    
    # Track which specific materials are in project BoMs
    materials_needed_detail = {}  # material_name -> monthly_consumption
    
    # Aggregate consumption across all active projects
    for project_id, project_info in appv6_projects.items():
        procurement_data = project_info['procurement_data']
        duration_months = project_info['procurement_duration']
        
        if 'monthly_schedule' not in procurement_data:
            continue
        
        # Process each material in the procurement schedule
        for material_schedule in procurement_data['monthly_schedule']:
            material_name = material_schedule.get('material', '')
            monthly_dist = material_schedule.get('monthly_distribution', {})
            
            # Track that this material is needed
            materials_in_active_projects.add(material_name.lower())
            
            # Map to warehouse category
            category = map_bom_material_to_warehouse_category(material_name)
            
            if not category or not monthly_dist:
                continue
            
            # Calculate average monthly consumption for this material
            total_qty = sum(monthly_dist.values())
            avg_monthly = total_qty / duration_months if duration_months > 0 else 0
            
            monthly_consumption_by_category[category] += avg_monthly
            
            # Track specific material consumption
            if material_name.lower() not in materials_needed_detail:
                materials_needed_detail[material_name.lower()] = 0
            materials_needed_detail[material_name.lower()] += avg_monthly
    
    # If no consumption data, return empty
    if all(v == 0 for v in monthly_consumption_by_category.values()):
        return pd.DataFrame(alerts)
    
    # Analyze each material in single warehouse
    for _, item in single_warehouse_df.iterrows():
        material_name = item['material_name']
        material_category = item['material_category']
        current_stock = item['current_stock_mt']
        capacity = item['capacity_mt']
        reorder_level = item['reorder_level_mt']
        safety_stock = item['safety_stock_mt']
        lead_time_days = item['lead_time_days']
        lead_time_months = lead_time_days / 30
        
        # Get monthly consumption for this category
        monthly_consumption = monthly_consumption_by_category.get(material_category, 0)
        
        # CHECK 1: Is this material actually needed by any active project?
        material_needed = False
        
        # Check if material name or category appears in project BoMs
        for needed_material in materials_in_active_projects:
            if (needed_material in material_name.lower() or 
                material_name.lower() in needed_material or
                material_category.lower() in needed_material):
                material_needed = True
                break
        
        # If material category has consumption, it's needed
        if monthly_consumption > 0:
            material_needed = True
        
        # SKIP ALERT: Material not needed by any active project
        if not material_needed:
            continue
        
        # Calculate metrics
        utilization_pct = (current_stock / capacity * 100) if capacity > 0 else 0
        months_remaining = current_stock / monthly_consumption if monthly_consumption > 0 else float('inf')
        
        # Thresholds for categorization
        SAFETY_BUFFER_MONTHS = 1
        MODERATE_OVERSTOCK_THRESHOLD = 6
        HIGH_OVERSTOCK_THRESHOLD = 9
        
        alert_type = None
        alert_icon = None
        alert_message = None
        priority = 0
        action_qty = 0
        severity = ""  # HIGH, MODERATE, LOW
        
        # === UNDERSTOCKING LOGIC ===
        
        # 1. CRITICAL: Below safety stock
        if current_stock < safety_stock:
            alert_type = "UNDERSTOCK"
            severity = "CRITICAL"
            alert_icon = "🔴"
            priority = 1
            target_stock = reorder_level + safety_stock
            action_qty = target_stock - current_stock
            alert_message = f"CRITICAL: Below safety stock! Current: {current_stock:.2f} MT, Safety: {safety_stock:.2f} MT"
        
        # 2. HIGH UNDERSTOCK: Will deplete within lead time (URGENT)
        elif months_remaining < lead_time_months:
            alert_type = "UNDERSTOCK"
            severity = "HIGH"
            alert_icon = "🔴"
            priority = 2
            target_stock = monthly_consumption * (lead_time_months + 2)  # Lead time + 2 months buffer
            action_qty = max(0, target_stock - current_stock)
            alert_message = f"HIGH PRIORITY: Stock depletes in {months_remaining:.1f} months, LESS than lead time ({lead_time_months:.1f} months)"
        
        # 3. MODERATE UNDERSTOCK: Will deplete within lead time + safety buffer
        elif months_remaining < (lead_time_months + SAFETY_BUFFER_MONTHS):
            alert_type = "UNDERSTOCK"
            severity = "MODERATE"
            alert_icon = "🟠"
            priority = 3
            target_stock = monthly_consumption * 4  # 4 months supply
            action_qty = max(0, target_stock - current_stock)
            alert_message = f"MODERATE: Stock depletes in {months_remaining:.1f} months. Order soon to maintain buffer."
        
        # 4. LOW UNDERSTOCK: At or near reorder level
        elif current_stock <= reorder_level:
            alert_type = "UNDERSTOCK"
            severity = "LOW"
            alert_icon = "🟡"
            priority = 4
            action_qty = (reorder_level + safety_stock) - current_stock
            alert_message = f"LOW: Stock at reorder level ({reorder_level:.2f} MT). Plan reorder."
        
        # === OVERSTOCKING LOGIC ===
        
        # 5. HIGH OVERSTOCK: Will last >9 months
        elif months_remaining > HIGH_OVERSTOCK_THRESHOLD:
            alert_type = "OVERSTOCK"
            severity = "HIGH"
            alert_icon = "🔵"
            priority = 6
            optimal_stock = monthly_consumption * 4
            action_qty = current_stock - optimal_stock
            alert_message = f"HIGH EXCESS: Stock will last {months_remaining:.1f} months (>9 months). Consider redistribution."
        
        # 6. MODERATE OVERSTOCK: Will last 6-9 months
        elif months_remaining > MODERATE_OVERSTOCK_THRESHOLD:
            alert_type = "OVERSTOCK"
            severity = "MODERATE"
            alert_icon = "🔵"
            priority = 5
            optimal_stock = monthly_consumption * 4
            action_qty = current_stock - optimal_stock
            alert_message = f"MODERATE EXCESS: Stock will last {months_remaining:.1f} months (6-9 months). Monitor future orders."
        
        # Only add actionable alerts
        if alert_type:
            alerts.append({
                'warehouse_name': item['warehouse_name'],
                'state': item['state'],
                'region': item['region'],
                'material_name': material_name,
                'material_category': material_category,
                'alert_type': alert_type,
                'severity': severity,
                'alert_icon': alert_icon,
                'priority': priority,
                'current_stock_mt': round(current_stock, 2),
                'capacity_mt': round(capacity, 2),
                'reorder_level_mt': round(reorder_level, 2),
                'safety_stock_mt': round(safety_stock, 2),
                'utilization_pct': round(utilization_pct, 2),
                'monthly_consumption_mt': round(monthly_consumption, 2),
                'months_remaining': round(months_remaining, 2) if months_remaining != float('inf') else 999,
                'lead_time_days': lead_time_days,
                'lead_time_months': round(lead_time_months, 2),
                'action_qty_mt': round(abs(action_qty), 2),
                'supplier_name': item['supplier_name'],
                'storage_location': item['storage_location'],
                'last_updated': item['last_updated'],
                'alert_message': alert_message
            })
    
    alerts_df = pd.DataFrame(alerts)
    
    # Sort by priority (understocking first), then by months remaining
    if len(alerts_df) > 0:
        alerts_df = alerts_df.sort_values(['priority', 'months_remaining'], ascending=[True, True])
    
    return alerts_df

# --- Data Update Functions ---
def update_warehouse_stock(warehouse_id, material, new_stock_value):
    """Update warehouse stock and save to CSV"""
    try:
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_path, 'Datasets', 'warehouse.csv')
        
        df = pd.read_csv(csv_path)
        
        material_lower = material.lower()
        stock_col = f'{material_lower}_current_stock_mt'
        
        mask = df['warehouse_id'] == warehouse_id
        
        if mask.any():
            df.loc[mask, stock_col] = new_stock_value
            df.to_csv(csv_path, index=False)
            st.cache_data.clear()
            return True, f"✅ Updated {material} stock for {warehouse_id} to {new_stock_value} MT"
        else:
            return False, f"❌ Warehouse {warehouse_id} not found"
            
    except Exception as e:
        return False, f"❌ Error updating warehouse: {str(e)}"

def update_vendor_price(vendor_id, new_price):
    """Update vendor price and save to CSV"""
    try:
        df = pd.read_csv('Datasets/vendors.csv')
        
        mask = df['vendor_id'] == vendor_id
        
        if mask.any():
            df.loc[mask, 'current_price_per_mt'] = new_price
            df.to_csv('Datasets/vendors.csv', index=False)
            st.cache_data.clear()
            return True, f"✅ Updated price for {vendor_id} to ₹{new_price}/MT"
        else:
            return False, f"❌ Vendor {vendor_id} not found"
            
    except Exception as e:
        return False, f"❌ Error updating vendor: {str(e)}"

def save_edited_projects(edited_df):
    """Save edited projects dataframe to CSV"""
    try:
        edited_df.to_csv('Datasets/powergrid_comprehensive_projects_bom_fixed_complete.csv', index=False)
        st.cache_data.clear()
        return True, "✅ Project data saved successfully"
    except Exception as e:
        return False, f"❌ Error saving projects: {str(e)}"

# --- Visualization Functions ---
def create_warehouse_capacity_chart(metrics_df, warehouse_name):
    """Create capacity utilization chart for a specific warehouse"""
    
    warehouse_data = metrics_df[metrics_df['warehouse_name'] == warehouse_name]
    
    fig = go.Figure()
    
    materials = warehouse_data['material'].tolist()
    stock = warehouse_data['current_stock_mt'].tolist()
    available = warehouse_data['available_capacity_mt'].tolist()
    
    fig.add_trace(go.Bar(
        name='Current Stock',
        x=materials,
        y=stock,
        marker_color='#667eea',
        text=[f'{s:.0f} MT' for s in stock],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Available Capacity',
        x=materials,
        y=available,
        marker_color='#c3cfe2',
        text=[f'{a:.0f} MT' for a in available],
        textposition='inside'
    ))
    
    fig.update_layout(
        title=f"Warehouse Capacity: {warehouse_name}",
        xaxis_title="Material Type",
        yaxis_title="Capacity (MT)",
        barmode='stack',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_utilization_heatmap(metrics_df):
    """Create heatmap of warehouse utilization across regions"""
    
    pivot_data = metrics_df.pivot_table(
        values='utilization_pct',
        index='warehouse_name',
        columns='material',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        text=pivot_data.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Utilization %")
    ))
    
    fig.update_layout(
        title="Warehouse Utilization Heatmap (% Capacity Used)",
        xaxis_title="Material Type",
        yaxis_title="Warehouse",
        height=600
    )
    
    return fig

def create_vendor_comparison_chart(vendors_df, material_category):
    """Create vendor price comparison chart"""
    
    category_vendors = vendors_df[vendors_df['material_category'] == material_category].sort_values('current_price_per_mt')
    
    if len(category_vendors) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=category_vendors['vendor_name'],
        y=category_vendors['current_price_per_mt'],
        marker_color=category_vendors['current_price_per_mt'],
        marker_colorscale='Viridis',
        text=[f"₹{p:,.0f}/MT" for p in category_vendors['current_price_per_mt']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.0f}/MT<br>Lead Time: %{customdata[0]} days<br>Discount: %{customdata[1]:.1f}%<extra></extra>',
        customdata=category_vendors[['lead_time_days', 'bulk_discount_pct']].values
    ))
    
    fig.update_layout(
        title=f"Vendor Price Comparison: {material_category}",
        xaxis_title="Vendor",
        yaxis_title="Price (₹/MT)",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_stock_depletion_chart(alerts_df):
    """Create visualization showing stock depletion timeline"""
    if alerts_df is None or len(alerts_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Create labels combining warehouse and material
    labels = [f"{row['warehouse_name']}<br>{row['material']}" 
              for _, row in alerts_df.iterrows()]
    
    months_remaining = alerts_df['months_remaining'].tolist()
    alert_types = alerts_df['alert_type'].tolist()
    
    # Color based on alert type
    colors = ['red' if at == 'UNDERSTOCKING' else 'orange' 
              for at in alert_types]
    
    fig.add_trace(go.Bar(
        y=labels,
        x=months_remaining,
        orientation='h',
        marker_color=colors,
        text=[f'{m:.1f} months' for m in months_remaining],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Stock lasts: %{x:.1f} months<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_vline(x=2, line_dash="dash", line_color="red", 
                  annotation_text="Lead Time (2 months)", 
                  annotation_position="top right")
    fig.add_vline(x=3, line_dash="dash", line_color="orange", 
                  annotation_text="Safety Threshold (3 months)", 
                  annotation_position="top right")
    fig.add_vline(x=6, line_dash="dash", line_color="green", 
                  annotation_text="Overstocking Threshold (6 months)", 
                  annotation_position="top right")
    
    fig.update_layout(
        title="Stock Depletion Timeline (Based on Current Consumption)",
        xaxis_title="Months Until Stock Depletes",
        yaxis_title="Warehouse & Material",
        height=max(400, len(alerts_df) * 30),
        showlegend=False,
        xaxis=dict(range=[0, max(months_remaining) * 1.1])
    )
    
    return fig

def create_single_warehouse_depletion_chart(alerts_df):
    """Create depletion timeline for single warehouse items"""
    if alerts_df is None or len(alerts_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Filter out infinite months
    chart_data = alerts_df[alerts_df['months_remaining'] < 999].copy()
    
    if len(chart_data) == 0:
        return None
    
    # Sort by months remaining
    chart_data = chart_data.sort_values('months_remaining')
    
    labels = chart_data['material_name'].tolist()
    months = chart_data['months_remaining'].tolist()
    alert_types = chart_data['alert_type'].tolist()
    
    # Color mapping
    color_map = {
        'CRITICAL_UNDERSTOCK': 'darkred',
        'UNDERSTOCKING': 'red',
        'REORDER_NEEDED': 'orange',
        'OVERSTOCKING': 'gold'
    }
    colors = [color_map.get(at, 'gray') for at in alert_types]
    
    fig.add_trace(go.Bar(
        y=labels,
        x=months,
        orientation='h',
        marker_color=colors,
        text=[f'{m:.1f} mo' for m in months],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Depletes in: %{x:.1f} months<extra></extra>'
    ))
    
    # Reference lines
    fig.add_vline(x=2, line_dash="dash", line_color="red", 
                  annotation_text="Avg Lead Time (2 mo)")
    fig.add_vline(x=3, line_dash="dash", line_color="orange", 
                  annotation_text="Safety Buffer (3 mo)")
    fig.add_vline(x=6, line_dash="dash", line_color="green", 
                  annotation_text="Overstock (6 mo)")
    
    fig.update_layout(
        title="Material Stock Depletion Timeline",
        xaxis_title="Months Until Stock Depletes",
        yaxis_title="Material",
        height=max(500, len(chart_data) * 25),
        showlegend=False
    )
    
    return fig

# --- Main Resource Management Interface ---
def show_resource_management():
    """Main function to display resource management interface"""
    
    st.title("📊 Resource Management Dashboard")
    st.markdown("**Real-time monitoring and management of warehouses, vendors, and projects**")
    
    # Load data
    projects_df = load_projects_data()
    vendors_df = load_vendors_data()
    warehouse_df = load_warehouse_data()
    
    # Calculate warehouse metrics
    if not warehouse_df.empty:
        warehouse_metrics = calculate_warehouse_metrics(warehouse_df)
    else:
        warehouse_metrics = pd.DataFrame()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Overview",
        "🏭 Warehouse Management",
        "🏢 Vendor Management",
        "📁 Project Database"
    ])
    
    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.header("📊 Real-Time Inventory Dashboard")
        
        # Load single warehouse data
        single_warehouse_df = load_single_warehouse_data()
        
        # Calculate alerts from single warehouse
        if not single_warehouse_df.empty and not projects_df.empty:
            # Load appV6 project data with procurement schedules
                appv6_projects = load_appv6_projects_data()

                if appv6_projects:
                    # Use actual procurement data from appV6
                    single_wh_alerts = calculate_single_warehouse_alerts_from_procurement(single_warehouse_df, appv6_projects)
                    st.info(f"📊 Using procurement data from {len(appv6_projects)} completed projects in appV6")
                else:
                    # Fallback to CSV-based calculation
                    single_wh_alerts = calculate_single_warehouse_alerts(single_warehouse_df, projects_df)
                    st.warning("⚠️ No appV6 project data found. Using CSV project data as fallback.")
        else:
            single_wh_alerts = pd.DataFrame()
        
        # Quick Stats Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Projects", len(projects_df))
        with col2:
            active_projects = len(projects_df[projects_df['Status'].isin(['Under Construction', 'Planned'])])
            st.metric("Active Projects", active_projects)
        with col3:
            if not single_warehouse_df.empty:
                st.metric("Warehouse Items", len(single_warehouse_df))
        with col4:
            if not single_wh_alerts.empty:
                critical_count = len(single_wh_alerts[single_wh_alerts['alert_type'].isin(['CRITICAL_UNDERSTOCK', 'UNDERSTOCKING', 'REORDER_NEEDED'])])
                st.metric("🔴 Action Required", critical_count, delta="URGENT" if critical_count > 0 else None)
        with col5:
            if not single_wh_alerts.empty:
                overstock_count = len(single_wh_alerts[single_wh_alerts['alert_type'] == 'OVERSTOCKING'])
                st.metric("🟡 Overstocked", overstock_count, delta="Review" if overstock_count > 0 else None)
        
        st.markdown("---")
        
        # Warehouse Inventory Status
        st.subheader("📦 PGCIL Central Warehouse - Inventory Status")
        
        if not single_wh_alerts.empty:
            # Show consumption summary
            with st.expander("📊 Current Consumption Rates from Active Projects", expanded=False):
                if appv6_projects:
                    st.success(f"✅ Using detailed procurement data from **{len(appv6_projects)}** active projects")
                    
                    # Show which projects are being analyzed
                    project_names = [p['project_name'] for p in appv6_projects.values()]
                    st.write("**Active Projects:**")
                    for i, name in enumerate(project_names, 1):
                        st.write(f"{i}. {name}")
                else:
                    st.info("Using aggregated data from CSV projects database")
            
            # Separate by severity levels
            critical_understock = single_wh_alerts[(single_wh_alerts['alert_type'] == 'UNDERSTOCK') & 
                                                   (single_wh_alerts['severity'] == 'CRITICAL')]
            high_understock = single_wh_alerts[(single_wh_alerts['alert_type'] == 'UNDERSTOCK') & 
                                               (single_wh_alerts['severity'] == 'HIGH')]
            moderate_understock = single_wh_alerts[(single_wh_alerts['alert_type'] == 'UNDERSTOCK') & 
                                                   (single_wh_alerts['severity'] == 'MODERATE')]
            low_understock = single_wh_alerts[(single_wh_alerts['alert_type'] == 'UNDERSTOCK') & 
                                              (single_wh_alerts['severity'] == 'LOW')]
            
            moderate_overstock = single_wh_alerts[(single_wh_alerts['alert_type'] == 'OVERSTOCK') & 
                                                  (single_wh_alerts['severity'] == 'MODERATE')]
            high_overstock = single_wh_alerts[(single_wh_alerts['alert_type'] == 'OVERSTOCK') & 
                                              (single_wh_alerts['severity'] == 'HIGH')]
            
            # Summary metrics row
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("🔴 CRITICAL", len(critical_understock), delta="URGENT!" if len(critical_understock) > 0 else None)
            with col2:
                st.metric("🔴 HIGH Under", len(high_understock), delta="Act Now" if len(high_understock) > 0 else None)
            with col3:
                st.metric("🟠 MOD Under", len(moderate_understock))
            with col4:
                st.metric("🟡 LOW Under", len(low_understock))
            with col5:
                st.metric("🔵 MOD Over", len(moderate_overstock))
            with col6:
                st.metric("🔵 HIGH Over", len(high_overstock))
            
            # Email Alert Button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📧 Alerts will be sent to: {ALERT_EMAIL}")
            with col2:
                if st.button("📧 Send Email Alert", type="primary", key="dashboard_email_single"):
                    with st.spinner("Sending email alert..."):
                        if send_stock_alert_email(single_wh_alerts):
                            st.success("✅ Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email")
            
            st.markdown("---")
            
            # === UNDERSTOCKING SECTION ===
            st.subheader("⚠️ UNDERSTOCKING ALERTS")
            
            # CRITICAL UNDERSTOCKING
            if len(critical_understock) > 0:
                st.error(f"### 🔴 CRITICAL: {len(critical_understock)} materials below safety stock - IMMEDIATE ACTION!")
                
                critical_display = critical_understock[[
                    'material_name', 'material_category', 'current_stock_mt', 'safety_stock_mt',
                    'monthly_consumption_mt', 'months_remaining', 'action_qty_mt', 'supplier_name'
                ]].copy()
                
                critical_display.columns = [
                    'Material', 'Category', 'Stock (MT)', 'Safety (MT)',
                    'Consumption/mo', 'Depletes (mo)', '🚨 ORDER NOW (MT)', 'Supplier'
                ]
                
                st.dataframe(critical_display, use_container_width=True, hide_index=True)
                
                with st.expander("💡 CRITICAL Actions"):
                    for _, row in critical_understock.iterrows():
                        st.error(f"**{row['material_name']}**: Order **{row['action_qty_mt']:.0f} MT** from {row['supplier_name']} IMMEDIATELY!")
            
            # HIGH UNDERSTOCKING
            if len(high_understock) > 0:
                st.error(f"### 🔴 HIGH PRIORITY: {len(high_understock)} materials depleting faster than lead time")
                
                high_display = high_understock[[
                    'material_name', 'material_category', 'current_stock_mt', 'monthly_consumption_mt',
                    'months_remaining', 'lead_time_months', 'action_qty_mt', 'supplier_name'
                ]].copy()
                
                high_display.columns = [
                    'Material', 'Category', 'Stock (MT)', 'Consumption/mo',
                    'Depletes (mo)', 'Lead Time (mo)', 'Order (MT)', 'Supplier'
                ]
                
                st.dataframe(high_display, use_container_width=True, hide_index=True)
                
                with st.expander("💡 HIGH Priority Actions"):
                    for _, row in high_understock.iterrows():
                        st.warning(f"**{row['material_name']}**: Order **{row['action_qty_mt']:.0f} MT** from {row['supplier_name']} within **{row['lead_time_days']:.0f} days**")
            
            # MODERATE UNDERSTOCKING
            if len(moderate_understock) > 0:
                st.warning(f"### 🟠 MODERATE PRIORITY: {len(moderate_understock)} materials need reordering soon")
                
                with st.expander("View Moderate Understocking Details", expanded=False):
                    moderate_display = moderate_understock[[
                        'material_name', 'current_stock_mt', 'monthly_consumption_mt',
                        'months_remaining', 'action_qty_mt', 'supplier_name'
                    ]].copy()
                    
                    moderate_display.columns = ['Material', 'Stock (MT)', 'Consumption/mo', 'Depletes (mo)', 'Order (MT)', 'Supplier']
                    st.dataframe(moderate_display, use_container_width=True, hide_index=True)
            
            # LOW UNDERSTOCKING
            if len(low_understock) > 0:
                st.info(f"### 🟡 LOW: {len(low_understock)} materials at reorder level")
                
                with st.expander("View Low Understocking Details", expanded=False):
                    low_display = low_understock[[
                        'material_name', 'current_stock_mt', 'reorder_level_mt', 'action_qty_mt', 'supplier_name'
                    ]].copy()
                    
                    low_display.columns = ['Material', 'Stock (MT)', 'Reorder Level (MT)', 'Order (MT)', 'Supplier']
                    st.dataframe(low_display, use_container_width=True, hide_index=True)
            
            # Success message if no understocking
            if len(critical_understock) == 0 and len(high_understock) == 0 and len(moderate_understock) == 0 and len(low_understock) == 0:
                st.success("✅ All materials have adequate stock levels - No understocking issues!")
            
            st.markdown("---")
            
            # === OVERSTOCKING SECTION ===
            st.subheader("📦 OVERSTOCKING ALERTS")
            
            # HIGH OVERSTOCKING
            if len(high_overstock) > 0:
                st.info(f"### 🔵 HIGH EXCESS: {len(high_overstock)} materials with >9 months supply")
                
                with st.expander("View High Overstocking Details", expanded=True):
                    high_over_display = high_overstock[[
                        'material_name', 'current_stock_mt', 'monthly_consumption_mt',
                        'months_remaining', 'action_qty_mt', 'storage_location'
                    ]].copy()
                    
                    high_over_display.columns = ['Material', 'Stock (MT)', 'Consumption/mo', 'Will Last (mo)', 'Excess (MT)', 'Location']
                    st.dataframe(high_over_display, use_container_width=True, hide_index=True)
                    
                    st.warning("💡 **Recommendations:** Consider redistributing excess stock to other warehouses or reducing future orders")
            
            # MODERATE OVERSTOCKING
            if len(moderate_overstock) > 0:
                st.info(f"### 🔵 MODERATE EXCESS: {len(moderate_overstock)} materials with 6-9 months supply")
                
                with st.expander("View Moderate Overstocking Details", expanded=False):
                    mod_over_display = moderate_overstock[[
                        'material_name', 'current_stock_mt', 'months_remaining', 'action_qty_mt', 'storage_location'
                    ]].copy()
                    
                    mod_over_display.columns = ['Material', 'Stock (MT)', 'Will Last (mo)', 'Excess (MT)', 'Location']
                    st.dataframe(mod_over_display, use_container_width=True, hide_index=True)
                    
                    st.info("💡 **Recommendations:** Monitor future procurement. No immediate action needed.")
            
            # Success message if no overstocking
            if len(moderate_overstock) == 0 and len(high_overstock) == 0:
                st.success("✅ No overstocking issues - Inventory levels are balanced!")
            
            st.markdown("---")
            
            # Stock Depletion Chart
            st.subheader("📈 Stock Depletion Timeline")
            fig_depletion = create_single_warehouse_depletion_chart(single_wh_alerts)
            if fig_depletion:
                st.plotly_chart(fig_depletion, use_container_width=True)

        else:
            st.info("ℹ️ No active projects or warehouse data. Load singlewarehouse.csv and ensure active projects exist.")
    
    # --- TAB 2: WAREHOUSE MANAGEMENT ---
    with tab2:
        st.header("🏭 Warehouse Management - Editable Inventory")
        st.info("💡 All stock values are editable. Edit directly in the table, then click 'Save Changes' to update inventory.")
        
        if warehouse_df.empty:
            st.error("❌ No warehouse data available")
        else:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_region_wh = st.multiselect(
                    "Region:",
                    warehouse_df['region'].unique(),
                    default=warehouse_df['region'].unique(),
                    key="wh_region_filter"
                )
            
            with col2:
                filter_state_wh = st.multiselect(
                    "State:",
                    warehouse_df['state'].unique(),
                    default=warehouse_df['state'].unique(),
                    key="wh_state_filter"
                )
            
            with col3:
                search_wh = st.text_input("🔍 Search Warehouse:", "", key="wh_search")
            
            # Apply filters
            filtered_warehouse = warehouse_df[
                (warehouse_df['region'].isin(filter_region_wh)) &
                (warehouse_df['state'].isin(filter_state_wh))
            ]
            
            if search_wh:
                filtered_warehouse = filtered_warehouse[
                    filtered_warehouse['warehouse_name'].str.contains(search_wh, case=False, na=False) |
                    filtered_warehouse['warehouse_id'].str.contains(search_wh, case=False, na=False)
                ]
            
            # Calculate metrics for filtered warehouses
            if not filtered_warehouse.empty:
                filtered_metrics = calculate_warehouse_metrics(filtered_warehouse)
                low_stock_in_view = len(filtered_metrics[filtered_metrics['status'] == 'LOW_STOCK'])
            else:
                low_stock_in_view = 0
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Warehouses", len(filtered_warehouse))
            with col2:
                total_steel = filtered_warehouse['steel_current_stock_mt'].sum()
                st.metric("Total Steel Stock", f"{total_steel:,.0f} MT")
            with col3:
                total_conductor = filtered_warehouse['conductor_current_stock_mt'].sum()
                st.metric("Total Conductor Stock", f"{total_conductor:,.0f} MT")
            with col4:
                st.metric("🟡 Low Stock Items", low_stock_in_view, delta="Action Required" if low_stock_in_view > 0 else None)
            
            st.markdown("---")
            
            # Editable Warehouse Table
            st.subheader("✏️ Editable Warehouse Inventory")
            st.warning("⚠️ Changes are temporary until you click 'Save Changes' button below")
            
            # Select which columns to display and edit
            display_columns = [
                'warehouse_id', 'warehouse_name', 'state', 'region',
                'steel_capacity_mt', 'steel_current_stock_mt',
                'conductor_capacity_mt', 'conductor_current_stock_mt',
                'insulator_capacity_mt', 'insulator_current_stock_mt',
                'concrete_capacity_mt', 'concrete_current_stock_mt',
                'hardware_capacity_mt', 'hardware_current_stock_mt'
            ]
            
            edited_warehouse_df = st.data_editor(
                filtered_warehouse[display_columns],
                use_container_width=True,
                hide_index=True,
                key="warehouse_editor",
                column_config={
                    "warehouse_id": st.column_config.TextColumn("Warehouse ID", disabled=True),
                    "warehouse_name": st.column_config.TextColumn("Warehouse Name", disabled=True),
                    "state": st.column_config.TextColumn("State", disabled=True),
                    "region": st.column_config.TextColumn("Region", disabled=True),
                    "steel_capacity_mt": st.column_config.NumberColumn(
                        "Steel Capacity (MT)",
                        disabled=True,
                        format="%.0f"
                    ),
                    "steel_current_stock_mt": st.column_config.NumberColumn(
                        "Steel Stock (MT)",
                        min_value=0,
                        format="%.2f",
                        help="Edit to update steel stock"
                    ),
                    "conductor_capacity_mt": st.column_config.NumberColumn(
                        "Conductor Capacity (MT)",
                        disabled=True,
                        format="%.0f"
                    ),
                    "conductor_current_stock_mt": st.column_config.NumberColumn(
                        "Conductor Stock (MT)",
                        min_value=0,
                        format="%.2f",
                        help="Edit to update conductor stock"
                    ),
                    "insulator_capacity_mt": st.column_config.NumberColumn(
                        "Insulator Capacity (MT)",
                        disabled=True,
                        format="%.0f"
                    ),
                    "insulator_current_stock_mt": st.column_config.NumberColumn(
                        "Insulator Stock (MT)",
                        min_value=0,
                        format="%.2f",
                        help="Edit to update insulator stock"
                    ),
                    "concrete_capacity_mt": st.column_config.NumberColumn(
                        "Concrete Capacity (MT)",
                        disabled=True,
                        format="%.0f"
                    ),
                    "concrete_current_stock_mt": st.column_config.NumberColumn(
                        "Concrete Stock (MT)",
                        min_value=0,
                        format="%.2f",
                        help="Edit to update concrete stock"
                    ),
                    "hardware_capacity_mt": st.column_config.NumberColumn(
                        "Hardware Capacity (MT)",
                        disabled=True,
                        format="%.0f"
                    ),
                    "hardware_current_stock_mt": st.column_config.NumberColumn(
                        "Hardware Stock (MT)",
                        min_value=0,
                        format="%.2f",
                        help="Edit to update hardware stock"
                    )
                }
            )
            
            st.markdown("---")
            
            # Save and Download buttons
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                if st.button("💾 Save Changes to Database", type="primary", use_container_width=True, key="save_wh_btn"):
                    # Merge edited data back into full warehouse dataframe
                    try:
                        # Load original full dataset
                        import os
                        base_path = os.path.dirname(os.path.abspath(__file__))
                        csv_path = os.path.join(base_path, 'Datasets', 'warehouse.csv')
                        full_warehouse_df = pd.read_csv(csv_path)
                        
                        # Update only the edited rows (match by warehouse_id)
                        for idx, row in edited_warehouse_df.iterrows():
                            wh_id = row['warehouse_id']
                            mask = full_warehouse_df['warehouse_id'] == wh_id
                            
                            if mask.any():
                                # Update stock columns
                                full_warehouse_df.loc[mask, 'steel_current_stock_mt'] = row['steel_current_stock_mt']
                                full_warehouse_df.loc[mask, 'conductor_current_stock_mt'] = row['conductor_current_stock_mt']
                                full_warehouse_df.loc[mask, 'insulator_current_stock_mt'] = row['insulator_current_stock_mt']
                                full_warehouse_df.loc[mask, 'concrete_current_stock_mt'] = row['concrete_current_stock_mt']
                                full_warehouse_df.loc[mask, 'hardware_current_stock_mt'] = row['hardware_current_stock_mt']
                        
                        # Save to CSV
                        full_warehouse_df.to_csv(csv_path, index=False)
                        st.cache_data.clear()
                        
                        st.success("✅ Warehouse inventory updated successfully!")
                        st.balloons()
                        
                        # Check for low stock items after update
                        updated_metrics = calculate_warehouse_metrics(full_warehouse_df)
                        low_stock_items = updated_metrics[updated_metrics['status'] == 'LOW_STOCK']
                        
                        if len(low_stock_items) > 0:
                            st.warning(f"⚠️ **{len(low_stock_items)} materials are now below 20% capacity!**")
                            
                            # Show low stock items
                            st.subheader("🟡 Low Stock Alert Details")
                            low_stock_display = low_stock_items[['status_icon', 'warehouse_name', 'state', 'material', 
                                                                 'current_stock_mt', 'capacity_mt', 'utilization_pct']]
                            low_stock_display = low_stock_display.sort_values('utilization_pct', ascending=True)
                            st.dataframe(low_stock_display, use_container_width=True, hide_index=True)
                            
                            # AUTOMATICALLY SEND EMAIL ALERT - NO CONFIRMATION NEEDED
                            st.info(f"📧 Sending low stock alert email to: {ALERT_EMAIL}...")
                            with st.spinner("Sending email alert..."):
                                if send_low_stock_email(low_stock_items):
                                    st.success(f"✅ Low stock alert email sent successfully to {ALERT_EMAIL}")
                                else:
                                    st.error("❌ Failed to send email alert. Check email configuration.")
                        else:
                            st.success("✅ All materials have adequate stock levels (>20%)")
                        
                    except Exception as e:
                        st.error(f"❌ Error saving warehouse data: {str(e)}")
            
            with col2:
                csv = edited_warehouse_df.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    f"warehouse_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🔄 Reset Changes", use_container_width=True, key="reset_wh_btn"):
                    st.rerun()
    
    # --- TAB 3: VENDOR MANAGEMENT ---
    with tab3:
        st.header("🏢 Vendor Management")
        
        if vendors_df.empty:
            st.error("❌ No vendor data available")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_category = st.selectbox(
                    "Material Category:",
                    ['All'] + list(vendors_df['material_category'].unique()),
                    key="vendor_category"
                )
            
            with col2:
                selected_state = st.selectbox(
                    "State:",
                    ['All'] + list(vendors_df['state'].unique()),
                    key="vendor_state"
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Sort By:",
                    ['current_price_per_mt', 'lead_time_days', 'bulk_discount_pct', 'capacity_mt_yr'],
                    key="vendor_sort"
                )
            
            filtered_vendors = vendors_df.copy()
            if selected_category != 'All':
                filtered_vendors = filtered_vendors[filtered_vendors['material_category'] == selected_category]
            if selected_state != 'All':
                filtered_vendors = filtered_vendors[filtered_vendors['state'] == selected_state]
            
            filtered_vendors = filtered_vendors.sort_values(sort_by)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Vendors", len(filtered_vendors))
            with col2:
                avg_price = filtered_vendors['current_price_per_mt'].mean()
                st.metric("Avg Price", f"₹{avg_price:,.0f}/MT")
            with col3:
                avg_lead_time = filtered_vendors['lead_time_days'].mean()
                st.metric("Avg Lead Time", f"{avg_lead_time:.0f} days")
            with col4:
                total_capacity = filtered_vendors['capacity_mt_yr'].sum()
                st.metric("Total Capacity", f"{total_capacity:,.0f} MT/yr")
            
            st.markdown("---")
            
            if selected_category != 'All':
                chart = create_vendor_comparison_chart(filtered_vendors, selected_category)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            st.subheader("📋 Vendor Directory")
            
            vendor_display = filtered_vendors[[
                'vendor_id', 'vendor_name', 'material_category', 'state',
                'current_price_per_mt', 'lead_time_days', 'bulk_discount_pct',
                'capacity_mt_yr', 'gst_pct'
            ]].copy()
            
            st.dataframe(vendor_display, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("💰 Update Vendor Pricing")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                vendor_to_update = st.selectbox(
                    "Select Vendor:",
                    filtered_vendors['vendor_id'].unique(),
                    key="vendor_update_select"
                )
            
            with col2:
                current_price = filtered_vendors[filtered_vendors['vendor_id'] == vendor_to_update]['current_price_per_mt'].values[0]
                
                new_price = st.number_input(
                    f"New Price (₹/MT) - Current: ₹{current_price:,.2f}",
                    min_value=0.0,
                    value=float(current_price),
                    step=100.0,
                    key="vendor_price_input"
                )
            
            with col3:
                st.write("")
                st.write("")
                if st.button("💾 Update Price", type="primary", key="vendor_update_btn"):
                    success, message = update_vendor_price(vendor_to_update, new_price)
                    if success:
                        st.success(message)
                        
                    else:
                        st.error(message)
    
    # --- TAB 4: PROJECT DATABASE (ALWAYS EDITABLE) ---
    with tab4:
        st.header("📁 Project Database - Editable Table")
        st.info("💡 All fields are editable. Edit values directly in the table, then click 'Save Changes'")
        
        if projects_df.empty:
            st.error("❌ No project data available")
        else:
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_region = st.multiselect(
                    "Region:",
                    projects_df['Region'].unique(),
                    default=projects_df['Region'].unique(),
                    key="proj_region_filter"
                )
            
            with col2:
                filter_voltage = st.multiselect(
                    "Voltage (kV):",
                    projects_df['Voltage_kV'].unique(),
                    default=projects_df['Voltage_kV'].unique(),
                    key="proj_voltage_filter"
                )
            
            with col3:
                filter_status = st.multiselect(
                    "Status:",
                    projects_df['Status'].unique(),
                    default=projects_df['Status'].unique(),
                    key="proj_status_filter"
                )
            
            with col4:
                filter_terrain = st.multiselect(
                    "Terrain:",
                    projects_df['Terrain_Type'].unique(),
                    default=projects_df['Terrain_Type'].unique(),
                    key="proj_terrain_filter"
                )
            
            # Apply filters
            filtered_projects = projects_df[
                (projects_df['Region'].isin(filter_region)) &
                (projects_df['Voltage_kV'].isin(filter_voltage)) &
                (projects_df['Status'].isin(filter_status)) &
                (projects_df['Terrain_Type'].isin(filter_terrain))
            ]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Projects Found", len(filtered_projects))
            with col2:
                total_steel = filtered_projects['Total_Steel_MT'].sum()
                st.metric("Total Steel", f"{total_steel:,.0f} MT")
            with col3:
                total_conductor = filtered_projects['Total_Conductor_MT'].sum()
                st.metric("Total Conductor", f"{total_conductor:,.0f} MT")
            with col4:
                total_concrete = filtered_projects['Total_Concrete_Cum'].sum()
                st.metric("Total Concrete", f"{total_concrete:,.0f} Cum")
            
            st.markdown("---")
            
            # Search
            search_term = st.text_input("🔍 Search projects by name or ID:", "", key="proj_search")
            
            if search_term:
                filtered_projects = filtered_projects[
                    filtered_projects['Project_Name'].str.contains(search_term, case=False, na=False) |
                    filtered_projects['Project_ID'].str.contains(search_term, case=False, na=False)
                ]
            
            # Editable Data Editor
            st.subheader("✏️ Editable Project Table")
            st.warning("⚠️ Changes are temporary until you click 'Save Changes' button below")
            
            edited_df = st.data_editor(
                filtered_projects,
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True,
                key="project_editor",
                column_config={
                    "Project_ID": st.column_config.TextColumn("Project ID", disabled=True),
                    "Status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["Commissioned", "Under Construction", "Planned", "Delayed"],
                        required=True
                    ),
                    "Completion_Year": st.column_config.NumberColumn(
                        "Completion Year",
                        min_value=2015,
                        max_value=2030,
                        step=1,
                        format="%d"
                    )
                }
            )
            
            st.markdown("---")
            
            # Save and Download buttons
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                if st.button("💾 Save Changes to Database", type="primary", use_container_width=True):
                    # Save edited dataframe
                    success, message = save_edited_projects(edited_df)
                    if success:
                        st.success(message)
                        #Removed rerun
                    else:
                        st.error(message)
            
            with col2:
                csv = edited_df.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    f"projects_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🔄 Reset Changes", use_container_width=True):
                    st.rerun()
    
    st.markdown("---")
    
    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⬅️ Back to Main App", type="secondary", use_container_width=True):
            st.session_state.show_resource_management = False
            st.rerun()

def load_appv6_projects_data():
    """
    Load project data from appV6.py session state
    Returns dictionary with project procurement schedules
    """
    if 'projects' not in st.session_state:
        return {}
    
    projects_with_procurement = {}
    
    for project_id, project_data in st.session_state.projects.items():
        # Only include completed projects with procurement data
        if (project_data.get('status') == 'complete' and 
            project_data.get('procurement_data') is not None and
            project_data.get('bom_df') is not None):
            
            projects_with_procurement[project_id] = {
                'project_name': project_data['project_inputs']['project_name'],
                'bom_df': project_data['bom_df'],
                'procurement_data': project_data['procurement_data'],
                'procurement_duration': project_data['project_inputs']['procurement_duration'],
                'region': project_data['project_inputs']['region'],
                'state': project_data['project_inputs']['state']
            }
    
    return projects_with_procurement

def map_bom_material_to_warehouse_category(material_name):
    """
    Map BoM material names to warehouse material categories
    """
    material_lower = material_name.lower()
    
    # Steel materials
    if any(keyword in material_lower for keyword in ['steel', 'tower', 'reinforcement', 'lattice', 'rebar']):
        return 'Steel'
    
    # Conductor materials
    elif any(keyword in material_lower for keyword in ['conductor', 'acsr', 'wire', 'cable', 'opgw']):
        return 'Conductor'
    
    # Insulator materials
    elif any(keyword in material_lower for keyword in ['insulator', 'disc', 'polymer']):
        return 'Insulator'
    
    # Concrete materials
    elif any(keyword in material_lower for keyword in ['concrete', 'foundation', 'cement']):
        return 'Concrete'
    
    # Hardware materials
    elif any(keyword in material_lower for keyword in ['bolt', 'nut', 'clamp', 'hardware', 'guard', 'plate', 'damper']):
        return 'Hardware'
    
    return None