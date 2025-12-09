from datetime import datetime
import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prompt_generator import (create_enhanced_bom_prompt,
                              create_procurement_analysis_prompt,
                              create_all_material_bom_prompts,
                              process_parallel_bom_generation,
                              create_all_procurement_prompts,
                              process_parallel_procurement_generation)
from tracker_module import show_resource_management
import PyPDF2
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="POWERGRID Demand Forecaster",
                   page_icon="⚡",
                   layout="wide")

# --- Gemini API Configuration ---
try:
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get(
        "GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
except (KeyError, AttributeError) as e:
    st.error(
        "🚨 Gemini API Key not found. Please add it to your Replit secrets or Streamlit secrets."
    )
    st.stop()

# --- POWERGRID Specific Data ---
TOWER_TYPES = {
    "132 kV Transmission Tower": {
        "base_materials":
        ["Steel Angle", "Galvanized Bolts", "Earth Wire", "Insulators"],
        "typical_steel_tons":
        15
    },
    "220 kV Transmission Tower": {
        "base_materials": [
            "Steel Angle", "Galvanized Bolts", "Earth Wire", "Insulators",
            "Phase Conductors"
        ],
        "typical_steel_tons":
        25
    },
    "400 kV Transmission Tower": {
        "base_materials": [
            "Steel Angle", "Galvanized Bolts", "Earth Wire", "Insulators",
            "Phase Conductors", "Bundle Conductors"
        ],
        "typical_steel_tons":
        45
    },
    "765 kV Transmission Tower": {
        "base_materials": [
            "Steel Angle", "Galvanized Bolts", "Earth Wire", "Insulators",
            "Phase Conductors", "Bundle Conductors"
        ],
        "typical_steel_tons":
        65
    }
}

SUBSTATION_TYPES = {
    "132/33 kV Substation": {
        "key_equipment": [
            "Power Transformer", "Circuit Breakers", "Isolators",
            "Current Transmitters", "Potential Transmitters"
        ],
        "typical_capacity":
        "50 MVA"
    },
    "220/132 kV Substation": {
        "key_equipment": [
            "Power Transformer", "Circuit Breakers", "Isolators",
            "Current Transmitters", "Potential Transmitters", "Surge Arresters"
        ],
        "typical_capacity":
        "100 MVA"
    },
    "400/220 kV Substation": {
        "key_equipment": [
            "Power Transformer", "Circuit Breakers", "Isolators",
            "Current Transmitters", "Potential Transmitters",
            "Surge Arresters", "Reactors"
        ],
        "typical_capacity":
        "315 MVA"
    },
    "765/400 kV Substation": {
        "key_equipment": [
            "Power Transformer", "Circuit Breakers", "Isolators",
            "Current Transmitters", "Potential Transmitters",
            "Surge Arresters", "Reactors"
        ],
        "typical_capacity":
        "500 MVA"
    }
}

REGIONS = {
    "Northern Region": {
        "states": [
            "Delhi", "Haryana", "Himachal Pradesh", "Jammu & Kashmir",
            "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand"
        ],
        "tax_multiplier":
        1.12
    },
    "Western Region": {
        "states": [
            "Gujarat", "Madhya Pradesh", "Maharashtra", "Chhattisgarh", "Goa",
            "Daman & Diu", "Dadra & Nagar Haveli"
        ],
        "tax_multiplier":
        1.15
    },
    "Southern Region": {
        "states": [
            "Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana",
            "Puducherry"
        ],
        "tax_multiplier":
        1.10
    },
    "Eastern Region": {
        "states": ["Bihar", "Jharkhand", "Odisha", "West Bengal", "Sikkim"],
        "tax_multiplier": 1.08
    },
    "North-Eastern Region": {
        "states": [
            "Assam", "Arunachal Pradesh", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Tripura"
        ],
        "tax_multiplier":
        1.05
    }
}

# --- STATIC MATERIAL PRICES (₹ per unit) ---
MATERIAL_PRICES = {
    "Galvanized Lattice Steel (Towers)": {"price": 75000, "unit": "MT", "gst": 18},
    "High-Tensile Bolts, Nuts, Washers": {"price": 150, "unit": "Nos", "gst": 18},
    "Concrete for Foundations (M20/25)": {"price": 6500, "unit": "Cum", "gst": 18},
    "Reinforcement Steel (Fe500/415)": {"price": 65000, "unit": "MT", "gst": 18},
    "ACSR Zebra Conductor": {"price": 320000, "unit": "MT", "gst": 18},
    "Earth Wire (GSS/OPGW)": {"price": 180000, "unit": "km", "gst": 18},
    "Suspension/Tension Insulators (Disc Type)": {"price": 450, "unit": "Nos", "gst": 18},
    "Insulator Hardware (Clamps, Sockets, Clevis)": {"price": 8500, "unit": "Sets", "gst": 18},
    "Grounding Rods/Electrodes (Cu/GS)": {"price": 2500, "unit": "Nos", "gst": 18},
    "Earthing Cable/Leads (Cu/GS)": {"price": 450, "unit": "M", "gst": 18},
    "Anti-Climbing Devices (Guards)": {"price": 1200, "unit": "Nos", "gst": 18},
    "Danger Plates/Warning Markers": {"price": 350, "unit": "Nos", "gst": 18},
    "Number/Phase/Circuit Plates": {"price": 300, "unit": "Nos", "gst": 18},
    "Bird Guards / Aviation Markers": {"price": 800, "unit": "Nos", "gst": 18},
}

# Add this helper function right after the MATERIAL_PRICES dictionary (around line 167)

def sort_months_chronologically(months):
    """Sort months in chronological order handling various formats"""
    def extract_month_number(month_str):
        """Extract numeric value from month string"""
        # Try to extract number from strings like "Month 1", "1", "Jan 2024", etc.
        import re
        match = re.search(r'\d+', str(month_str))
        if match:
            return int(match.group())
        return 0
    
    return sorted(months, key=extract_month_number)


# Update the three chart functions to use this sorting:

def create_procurement_timeline_chart(procurement_data, selected_material=None):
    """Create interactive procurement timeline chart showing material distribution"""
    if not procurement_data or 'monthly_schedule' not in procurement_data:
        return None
    
    schedule = procurement_data['monthly_schedule']
    
    # Prepare data
    all_months = set()
    materials_data = {}
    
    for material_schedule in schedule:
        material_name = material_schedule.get('material', 'Unknown')
        monthly_dist = material_schedule.get('monthly_distribution', {})
        
        if not monthly_dist:
            continue
        
        # Filter by selected material if specified
        if selected_material and material_name != selected_material:
            continue
        
        materials_data[material_name] = monthly_dist
        all_months.update(monthly_dist.keys())
    
    if not materials_data:
        return None
    
    # Sort months chronologically - FIXED
    sorted_months = sort_months_chronologically(list(all_months))
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    for idx, (material_name, monthly_dist) in enumerate(materials_data.items()):
        quantities = [monthly_dist.get(month, 0) for month in sorted_months]
        
        fig.add_trace(go.Scatter(
            x=sorted_months,
            y=quantities,
            mode='lines+markers',
            name=material_name,
            line=dict(width=3, color=colors[idx % len(colors)]),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Quantity: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    title = f"Procurement Timeline - {selected_material}" if selected_material else "Procurement Timeline - All Materials"
    
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Quantity",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis=dict(tickangle=-45)
    )
    
    return fig


def create_procurement_stacked_area_chart(procurement_data):
    """Create stacked area chart for procurement schedule"""
    if not procurement_data or 'monthly_schedule' not in procurement_data:
        return None
    
    schedule = procurement_data['monthly_schedule']
    
    # Prepare data
    all_months = set()
    materials_data = {}
    
    for material_schedule in schedule:
        material_name = material_schedule.get('material', 'Unknown')
        monthly_dist = material_schedule.get('monthly_distribution', {})
        
        if not monthly_dist:
            continue
        
        materials_data[material_name] = monthly_dist
        all_months.update(monthly_dist.keys())
    
    if not materials_data:
        return None
    
    # Sort months chronologically - FIXED
    sorted_months = sort_months_chronologically(list(all_months))
    
    # Create DataFrame for easier plotting
    data = {'Month': sorted_months}
    for material_name, monthly_dist in materials_data.items():
        data[material_name] = [monthly_dist.get(month, 0) for month in sorted_months]
    
    df = pd.DataFrame(data)
    
    # Create stacked area chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    for idx, material in enumerate([col for col in df.columns if col != 'Month']):
        fig.add_trace(go.Scatter(
            x=df['Month'],
            y=df[material],
            mode='lines',
            name=material,
            stackgroup='one',
            fillcolor=colors[idx % len(colors)],
            line=dict(width=0.5, color=colors[idx % len(colors)]),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Month: %{x}<br>' +
                         'Quantity: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="Cumulative Procurement Schedule - Stacked View",
        xaxis_title="Month",
        yaxis_title="Total Quantity",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis=dict(tickangle=-45)
    )
    
    return fig


def create_material_heatmap(procurement_data):
    """Create heatmap showing procurement intensity by material and month"""
    if not procurement_data or 'monthly_schedule' not in procurement_data:
        return None
    
    schedule = procurement_data['monthly_schedule']
    
    # Prepare data
    all_months = set()
    materials_data = {}
    
    for material_schedule in schedule:
        material_name = material_schedule.get('material', 'Unknown')
        monthly_dist = material_schedule.get('monthly_distribution', {})
        
        if not monthly_dist:
            continue
        
        materials_data[material_name] = monthly_dist
        all_months.update(monthly_dist.keys())
    
    if not materials_data:
        return None
    
    # Sort months and materials - FIXED
    sorted_months = sort_months_chronologically(list(all_months))
    sorted_materials = sorted(materials_data.keys())
    
    # Create matrix
    matrix = []
    for material in sorted_materials:
        row = [materials_data[material].get(month, 0) for month in sorted_months]
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=sorted_months,
        y=sorted_materials,
        colorscale='YlOrRd',
        hovertemplate='Material: %{y}<br>Month: %{x}<br>Quantity: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Procurement Intensity Heatmap",
        xaxis_title="Month",
        yaxis_title="Material",
        height=max(400, len(sorted_materials) * 30),
        xaxis=dict(tickangle=-45)
    )
    
    return fig
# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def parse_bom_from_text(text):
    """Parse BoM from extracted text using AI"""
    try:
        prompt = f"""Extract a Bill of Materials table from this text. Return ONLY a JSON array with this structure:
[{{"material_name": "...", "quantity": number, "unit": "..."}}]

Text:
{text[:5000]}

Return ONLY valid JSON, no other text."""
        
        response = model.generate_content(prompt)
        json_data = parse_json_from_text(response.text)
        return json_data
    except Exception as e:
        st.error(f"Error parsing BoM: {str(e)}")
        return None

def compare_boms(uploaded_bom, generated_bom):
    """Compare uploaded BoM with generated BoM"""
    comparison = []
    
    # Create lookup for generated BoM
    gen_lookup = {row['Material_Name'].lower(): row for _, row in generated_bom.iterrows()}
    
    for item in uploaded_bom:
        material_name = item['material_name']
        uploaded_qty = item['quantity']
        unit = item['unit']
        
        match = gen_lookup.get(material_name.lower())
        
        if match:
            generated_qty = match['Estimated_Quantity']
            difference = abs(generated_qty - uploaded_qty)
            diff_pct = (difference / uploaded_qty * 100) if uploaded_qty > 0 else 0
            
            status = "✅ Match" if diff_pct < 10 else "⚠️ Deviation" if diff_pct < 30 else "🔴 Major Deviation"
            
            comparison.append({
                'Material': material_name,
                'Uploaded_Qty': uploaded_qty,
                'Generated_Qty': generated_qty,
                'Difference': difference,
                'Diff_%': diff_pct,
                'Unit': unit,
                'Status': status
            })
        else:
            comparison.append({
                'Material': material_name,
                'Uploaded_Qty': uploaded_qty,
                'Generated_Qty': 0,
                'Difference': uploaded_qty,
                'Diff_%': 100,
                'Unit': unit,
                'Status': "❌ Missing"
            })
    
    # Check for materials in generated but not in uploaded
    for _, row in generated_bom.iterrows():
        if row['Material_Name'].lower() not in [item['material_name'].lower() for item in uploaded_bom]:
            comparison.append({
                'Material': row['Material_Name'],
                'Uploaded_Qty': 0,
                'Generated_Qty': row['Estimated_Quantity'],
                'Difference': row['Estimated_Quantity'],
                'Diff_%': 100,
                'Unit': row['Unit'],
                'Status': "➕ Additional"
            })
    
    return pd.DataFrame(comparison)

def calculate_static_costs(bom_df, region):
    """Calculate costs using static pricing"""
    material_costs = []
    total_cost = 0
    gst_summary = []
    
    tax_multiplier = REGIONS[region]['tax_multiplier']
    
    for _, row in bom_df.iterrows():
        material_name = row['Material_Name']
        quantity = float(row['Estimated_Quantity'])
        unit = row['Unit']
        
        # Find matching price
        price_info = None
        for key, val in MATERIAL_PRICES.items():
            if key.lower() in material_name.lower() or material_name.lower() in key.lower():
                price_info = val
                break
        
        if not price_info:
            # Default pricing if not found
            price_info = {"price": 10000, "unit": unit, "gst": 18}
        
        base_cost = quantity * price_info['price']
        gst_amount = base_cost * (price_info['gst'] / 100)
        regional_cost = (base_cost + gst_amount) * tax_multiplier
        
        material_costs.append({
            'Material': material_name,
            'Quantity': quantity,
            'Unit': unit,
            'Unit_Price': price_info['price'],
            'Base_Cost': base_cost / 10000000,  # Convert to Crores
            'GST_%': price_info['gst'],
            'GST_Amount': gst_amount / 10000000,
            'Regional_Cost': regional_cost / 10000000,
        })
        
        total_cost += regional_cost / 10000000
    
    return {
        'total_cost': total_cost,
        'material_costs': material_costs,
        'gst_summary': gst_summary,
        'region': region,
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def parse_markdown_table(markdown_string):
    """Parse markdown table - NO FALLBACKS"""
    if not markdown_string:
        return None

    cleaned_string = markdown_string.replace('\\_', '_').replace('\\|', '|')
    lines = cleaned_string.strip().split('\n')
    table_lines = [
        line.strip() for line in lines if line.strip().startswith('|')
    ]

    if len(table_lines) < 2:
        return None

    try:
        headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
        rows = []
        for line in table_lines[2:]:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
        
        return pd.DataFrame(rows) if rows else None

    except Exception as e:
        return None


def parse_json_from_text(text):
    """Extract JSON from AI response - IMPROVED VERSION"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Method 2: Extract from markdown code blocks
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Method 3: Find largest JSON object in text
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            return json.loads(json_str)
    except Exception as e:
        pass

    return None


def calculate_monthly_costs_from_procurement(procurement_data,
                                             material_costs_dict):
    """
    Calculate monthly costs based on procurement schedule and material prices
    """
    monthly_costs = {}

    if not procurement_data or 'monthly_schedule' not in procurement_data:
        return {}

    # Create material name to cost mapping
    cost_lookup = {}
    for item in material_costs_dict:
        material_name = item['Material'].lower()
        unit_price = item['Unit_Price']
        cost_lookup[material_name] = unit_price

    # Process each material's monthly distribution
    for material_schedule in procurement_data['monthly_schedule']:
        material_name = material_schedule.get('material', '').lower()
        
        if 'monthly_distribution' not in material_schedule:
            continue
            
        monthly_dist = material_schedule['monthly_distribution']
        
        # Find unit price
        unit_price = None
        for key in cost_lookup:
            if key in material_name or material_name in key:
                unit_price = cost_lookup[key]
                break
        
        if not unit_price:
            continue
        
        for month, qty in monthly_dist.items():
            if month not in monthly_costs:
                monthly_costs[month] = 0
            monthly_costs[month] += (qty * unit_price) / 10000000  # Convert to Crores

    # Round values
    for month in monthly_costs:
        monthly_costs[month] = round(monthly_costs[month], 2)

    return monthly_costs


def create_cost_breakdown_chart(cost_data):
    """Create cost breakdown visualization"""
    if not cost_data:
        return None

    materials = []
    costs = []

    for item in cost_data:
        materials.append(item['Material'])
        costs.append(item['Regional_Cost'])

    fig = px.pie(values=costs,
                 names=materials,
                 title="Material Cost Distribution",
                 hole=0.4)

    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig


def create_monthly_cost_chart(monthly_data):
    """Create monthly cost analysis chart"""
    if not monthly_data:
        return None

    months = list(monthly_data.keys())
    costs = list(monthly_data.values())

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=months,
               y=costs,
               name='Monthly Cost',
               marker_color='#FF6B6B',
               text=[f'₹{c:.2f}Cr' for c in costs],
               textposition='outside'))

    # Add cumulative line
    cumulative = np.cumsum(costs)
    fig.add_trace(
        go.Scatter(x=months,
                   y=cumulative,
                   name='Cumulative Cost',
                   mode='lines+markers',
                   marker_color='#4ECDC4',
                   line=dict(width=3),
                   yaxis='y2'))

    fig.update_layout(
        title="Monthly Cost Analysis (Based on Procurement Schedule)",
        xaxis_title="Month",
        yaxis_title="Monthly Cost (₹ Crores)",
        yaxis2=dict(title="Cumulative Cost (₹ Crores)",
                    overlaying='y',
                    side='right'),
        hovermode='x unified',
        height=500)

    return fig


# --- Session State Initialization ---
if 'projects' not in st.session_state:
    st.session_state.projects = {}
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None
if 'show_resource_management' not in st.session_state:
    st.session_state.show_resource_management = False

# --- Main Application ---

# Check if resource management should be shown
if st.session_state.get('show_resource_management', False):
    show_resource_management()
    st.stop()

st.title("⚡ POWERGRID AI Demand & Procurement Planner")
st.markdown(
    "**Multi-Project Planning System with AI-Powered Demand Forecasting**"
)

# --- Top Navigation ---
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    if len(st.session_state.projects) > 0:
        project_names = {pid: pdata['project_inputs']['project_name'] 
                        for pid, pdata in st.session_state.projects.items()}
        selected_project = st.selectbox(
            "📁 Select Project",
            options=['[Create New]'] + list(project_names.keys()),
            format_func=lambda x: x if x == '[Create New]' else project_names[x],
            key='project_selector'
        )
        
        if selected_project != '[Create New]':
            st.session_state.current_project_id = selected_project
    else:
        st.info("No projects yet. Create your first project below.")
        st.session_state.current_project_id = None

with col2:
    if st.button("➕ Create New Project", use_container_width=True):
        st.session_state.current_project_id = None
        st.rerun()

with col3:
    if st.button("📊 Resources", type="secondary", use_container_width=True):
        st.session_state.show_resource_management = True
        st.rerun()

st.markdown("---")

# --- Project Management Section ---
current_project = None
if st.session_state.current_project_id:
    current_project = st.session_state.projects.get(st.session_state.current_project_id)

# --- Sidebar: Project Configuration ---
with st.sidebar:
    st.header("🏗️ Project Configuration")
    
    # Load values from current project if editing
    default_values = current_project['project_inputs'] if current_project else {}

    project_name = st.text_input("Project Name", 
                                 value=default_values.get('project_name', 'POWERGRID Transmission Project'))
    project_type = st.selectbox(
        "Project Type",
        list(TOWER_TYPES.keys()) + list(SUBSTATION_TYPES.keys()),
        index=0 if not default_values else 
              (list(TOWER_TYPES.keys()) + list(SUBSTATION_TYPES.keys())).index(default_values.get('project_type', list(TOWER_TYPES.keys())[0]))
    )

    region = st.selectbox("Region", list(REGIONS.keys()),
                         index=0 if not default_values else list(REGIONS.keys()).index(default_values.get('region', list(REGIONS.keys())[0])))
    state = st.selectbox("State", REGIONS[region]["states"])
    terrain = st.selectbox("Terrain Type",
                           ["Plain", "Hilly", "Desert", "Coastal", "Forest"],
                           index=0 if not default_values else ["Plain", "Hilly", "Desert", "Coastal", "Forest"].index(default_values.get('terrain', 'Plain')))

    st.markdown("---")
    st.header("📊 Project Specifications")

    if "Tower" in project_type:
        line_length = st.number_input("Line Length (km)",
                                     min_value=10,
                                     max_value=500,
                                     value=default_values.get('line_length', 100),
                                     step=10)
        capacity_mw = st.number_input("Capacity (MW)",
                                     min_value=100,
                                     max_value=5000,
                                     value=default_values.get('capacity_mw', 1000),
                                     step=100)
        num_bays = 0
        substation_capacity = 0
    else:
        num_bays = st.number_input("Number of Bays",
                                   min_value=2,
                                   max_value=20,
                                   value=default_values.get('num_bays', 6),
                                   step=1)
        substation_capacity = st.number_input(
            "Substation Capacity (MVA)",
            min_value=50,
            max_value=1000,
            value=default_values.get('substation_capacity', 315),
            step=50)
        line_length = 0
        capacity_mw = 0

    st.markdown("---")
    st.header("💰 Financial Parameters")

    total_budget = st.number_input("Total Project Budget (Crores INR)",
                                   min_value=10,
                                   max_value=5000,
                                   value=default_values.get('total_budget', 500),
                                   step=50)

    st.markdown("---")
    st.header("⏱️ Timeline")

    procurement_duration = st.slider("Procurement Duration (Months)",
                                     min_value=6,
                                     max_value=48,
                                     value=default_values.get('procurement_duration', 18))

    st.markdown("---")
    st.header("⚙️ Advanced Parameters")

    monsoon_factor = st.checkbox("Consider Monsoon Delays", 
                                 value=default_values.get('monsoon_factor', True))
    
    st.markdown("---")
    st.header("📄 Optional: Upload Existing BoM")
    
    uploaded_bom_file = st.file_uploader("Upload BoM PDF (Optional)", 
                                        type=['pdf'],
                                        help="Upload existing BoM for comparison")

# --- Main Content Area ---
if not st.session_state.current_project_id:
    # NEW PROJECT CREATION
    st.subheader("➕ Create New Project")
    
    with st.expander("📋 Project Summary Preview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Project:** {project_name}")
            st.write(f"**Type:** {project_type}")
            st.write(f"**Location:** {state}, {region}")
            st.write(f"**Terrain:** {terrain}")
        with col2:
            if "Tower" in project_type:
                st.write(f"**Line Length:** {line_length} km")
                st.write(f"**Capacity:** {capacity_mw} MW")
            else:
                st.write(f"**Bays:** {num_bays}")
                st.write(f"**Capacity:** {substation_capacity} MVA")
            st.write(f"**Budget:** ₹{total_budget} Crores")
            st.write(f"**Duration:** {procurement_duration} months")
    
    if st.button("💾 Save and Add Project", type="primary", use_container_width=True):
        # Create new project
        project_id = f"PRJ_{len(st.session_state.projects) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        project_data = {
            'project_inputs': {
                'project_name': project_name,
                'project_type': project_type,
                'region': region,
                'state': state,
                'terrain': terrain,
                'line_length': line_length,
                'capacity_mw': capacity_mw,
                'num_bays': num_bays,
                'substation_capacity': substation_capacity,
                'total_budget': total_budget,
                'procurement_duration': procurement_duration,
                'monsoon_factor': monsoon_factor
            },
            'status': 'configured',
            'bom_df': None,
            'cost_data': None,
            'procurement_data': None,
            'uploaded_bom': None,
            'comparison_df': None,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.projects[project_id] = project_data
        st.session_state.current_project_id = project_id
        st.success(f"✅ Project '{project_name}' created successfully!")
        st.rerun()

else:
    # EXISTING PROJECT - SHOW WORKFLOW
    current_project = st.session_state.projects[st.session_state.current_project_id]
    pi = current_project['project_inputs']
    
    # Step indicators
    status = current_project['status']
    step_map = {
        'configured': "1️⃣ Ready to Generate BoM",
        'bom_generated': "2️⃣ BoM Generated - Ready for Cost Analysis",
        'costs_calculated': "3️⃣ Costs Calculated - Ready for Procurement",
        'complete': "4️⃣ Complete - View Reports"
    }
    
    st.info(f"**Current Status:** {step_map.get(status, 'Unknown')}")
    
    # Step 1: Generate BoM
    if status == 'configured':
        st.subheader("Step 1: Generate Bill of Materials")
        
        with st.expander("📋 Project Summary", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Project:** {pi['project_name']}")
                st.write(f"**Type:** {pi['project_type']}")
                st.write(f"**Location:** {pi['state']}, {pi['region']}")
                st.write(f"**Terrain:** {pi['terrain']}")
            with col2:
                if "Tower" in pi['project_type']:
                    st.write(f"**Line Length:** {pi['line_length']} km")
                    st.write(f"**Capacity:** {pi['capacity_mw']} MW")
                else:
                    st.write(f"**Bays:** {pi['num_bays']}")
                    st.write(f"**Capacity:** {pi['substation_capacity']} MVA")
                st.write(f"**Budget:** ₹{pi['total_budget']} Crores")
                st.write(f"**Duration:** {pi['procurement_duration']} months")
        
        # Handle uploaded BoM
        if uploaded_bom_file:
            st.info("📄 BoM PDF uploaded - Will compare after generation")
            pdf_text = extract_text_from_pdf(uploaded_bom_file)
            if pdf_text:
                uploaded_bom_data = parse_bom_from_text(pdf_text)
                if uploaded_bom_data:
                    current_project['uploaded_bom'] = uploaded_bom_data
                    st.success(f"✅ Extracted {len(uploaded_bom_data)} items from uploaded BoM")
        
        if st.button("🚀 Generate Baseline BoM", type="primary", use_container_width=True):
            with st.spinner("🧠 AI generating BoM using POWERGRID standards (Parallel Processing)..."):
                # Create individual prompts for each material
                material_prompts = create_all_material_bom_prompts(pi)
                
                st.info(f"🚀 Processing {len(material_prompts)} materials in parallel...")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, material_name, success):
                    progress_bar.progress(current / total)
                    status_text.text(
                        f"Processing {current}/{total}: {material_name} {'✅' if success else '❌'}"
                    )
                
                try:
                    # Process all materials in parallel
                    results = process_parallel_bom_generation(
                        material_prompts,
                        model,
                        max_workers=20,
                        progress_callback=update_progress)
                    
                    # Convert results to DataFrame
                    bom_data = []
                    success_count = 0
                    for r in results:
                        if r['success']:
                            success_count += 1
                        data = r['data']
                        bom_data.append({
                            'Material_Name': data.get('material_name', ''),
                            'Unit': data.get('unit', ''),
                            'Estimated_Quantity': float(data.get('estimated_quantity', 0)),
                            'Priority': data.get('priority', 'Medium'),
                            'Calculation_Method': data.get('calculation_method', ''),
                            'Notes': data.get('notes', '')
                        })
                    
                    bom_df = pd.DataFrame(bom_data)
                    
                    if len(bom_df) > 0:
                        current_project['bom_df'] = bom_df
                        current_project['status'] = 'bom_generated'
                        
                        # If uploaded BoM exists, do comparison
                        if current_project.get('uploaded_bom'):
                            comparison_df = compare_boms(current_project['uploaded_bom'], bom_df)
                            current_project['comparison_df'] = comparison_df
                        
                        st.success(f"✅ BoM generated successfully! ({success_count}/{len(results)} materials processed)")
                        progress_bar.empty()
                        status_text.empty()
                        st.rerun()
                    else:
                        st.error("❌ AI failed to generate valid BoM. Please try again.")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Step 2: Review BoM and Calculate Costs
    elif status == 'bom_generated':
        st.subheader("Step 2: Review BoM and Generate Cost Analysis")
        
        # Show comparison if available
        if current_project.get('comparison_df') is not None:
            st.warning("📊 BoM Comparison with Uploaded Document")
            
            comp_df = current_project['comparison_df']
            
            # Color code the comparison
            def highlight_status(row):
                if row['Status'] == '✅ Match':
                    return ['background-color: #90EE90'] * len(row)
                elif row['Status'] == '⚠️ Deviation':
                    return ['background-color: #FFD700'] * len(row)
                elif row['Status'] == '🔴 Major Deviation':
                    return ['background-color: #FFB6C6'] * len(row)
                elif row['Status'] == '❌ Missing':
                    return ['background-color: #FF6B6B'] * len(row)
                elif row['Status'] == '➕ Additional':
                    return ['background-color: #87CEEB'] * len(row)
                return [''] * len(row)
            
            styled_comp = comp_df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_comp, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                matches = len(comp_df[comp_df['Status'] == '✅ Match'])
                st.metric("Matches", matches)
            with col2:
                deviations = len(comp_df[comp_df['Status'].str.contains('Deviation')])
                st.metric("Deviations", deviations)
            with col3:
                missing = len(comp_df[comp_df['Status'] == '❌ Missing'])
                st.metric("Missing", missing)
            with col4:
                additional = len(comp_df[comp_df['Status'] == '➕ Additional'])
                st.metric("Additional", additional)
        
        # Editable BoM
        st.markdown("### 📝 Edit BoM (if needed)")
        edited_df = st.data_editor(current_project['bom_df'],
                                   num_rows="dynamic",
                                   use_container_width=True,
                                   column_config={
                                       "Priority": st.column_config.SelectboxColumn(
                                           "Priority",
                                           help="Material priority",
                                           width="medium",
                                           options=["High", "Medium", "Low"],
                                           required=True,
                                       )
                                   })
        
        if st.button("💰 Calculate Costs & Generate Procurement", type="primary", use_container_width=True):
            current_project['bom_df'] = edited_df
            
            with st.spinner("💰 Calculating costs using static pricing..."):
                try:
                    cost_data = calculate_static_costs(edited_df, pi['region'])
                    current_project['cost_data'] = cost_data
                    st.success("✅ Costs calculated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Cost calculation failed: {str(e)}")
                    st.stop()
            
            with st.spinner("📅 Generating procurement schedule (Parallel Processing)..."):
                try:
                    procurement_prompts = create_all_procurement_prompts(edited_df, pi)
                    
                    st.info(f"🚀 Processing {len(procurement_prompts)} material schedules in parallel...")
                    
                    proc_progress_bar = st.progress(0)
                    proc_status_text = st.empty()
                    
                    def update_proc_progress(current, total, material_name, success):
                        proc_progress_bar.progress(current / total)
                        proc_status_text.text(
                            f"Scheduling {current}/{total}: {material_name} {'✅' if success else '❌'}"
                        )
                    
                    procurement_json = process_parallel_procurement_generation(
                        procurement_prompts,
                        model,
                        max_workers=20,
                        progress_callback=update_proc_progress)
                    
                    proc_progress_bar.empty()
                    proc_status_text.empty()
                    
                    if procurement_json and 'monthly_schedule' in procurement_json:
                        current_project['procurement_data'] = procurement_json
                        st.success(f"✅ Procurement schedule generated! ({len(procurement_json['monthly_schedule'])} materials)")
                    else:
                        st.error("❌ Failed to generate procurement schedule")
                        st.stop()
                
                except Exception as e:
                    st.error(f"❌ Procurement generation failed: {str(e)}")
                    st.stop()
            
            with st.spinner("📊 Calculating monthly costs..."):
                try:
                    monthly_costs = calculate_monthly_costs_from_procurement(
                        current_project['procurement_data'],
                        current_project['cost_data']['material_costs'])
                    
                    current_project['cost_data']['monthly_costs'] = monthly_costs
                    current_project['status'] = 'complete'
                    
                    st.success("✅ All analysis completed!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Monthly cost calculation failed: {str(e)}")
                    current_project['status'] = 'complete'
                    st.rerun()
    
    # Step 3: View Complete Reports
    elif status == 'complete':
        st.subheader("📊 Complete Project Analysis")
        
        cost_data = current_project['cost_data']
        procurement_data = current_project['procurement_data']
        bom_df = current_project['bom_df']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_cost = cost_data.get('total_cost', 0)
            st.metric("Total Material Cost", f"₹{total_cost:.2f} Cr")
        with col2:
            st.metric("Project Budget", f"₹{pi['total_budget']} Cr")
        with col3:
            budget_utilization = (total_cost / pi['total_budget']) * 100 if pi['total_budget'] > 0 else 0
            st.metric("Budget Utilization", f"{budget_utilization:.1f}%")
        with col4:
            st.metric("Procurement Duration", f"{pi['procurement_duration']} months")
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Cost Analysis", "📅 Procurement Schedule", "📋 BoM", "📊 Comparison", "📄 Report"
        ])
        
        with tab1:
            st.header("💰 Detailed Cost Analysis")
            
            if 'material_costs' in cost_data:
                cost_df = pd.DataFrame(cost_data['material_costs'])
                st.dataframe(cost_df, use_container_width=True)
                
                if len(cost_df) > 0:
                    fig_cost = create_cost_breakdown_chart(cost_data['material_costs'])
                    if fig_cost:
                        st.plotly_chart(fig_cost, use_container_width=True)
            
            if 'monthly_costs' in cost_data and cost_data['monthly_costs']:
                st.subheader("📈 Monthly Cost Analysis")
                fig_monthly = create_monthly_cost_chart(cost_data['monthly_costs'])
                if fig_monthly:
                    st.plotly_chart(fig_monthly, use_container_width=True)
                
                with st.expander("📊 View Monthly Cost Breakdown Table"):
                    monthly_df = pd.DataFrame({
                        'Month': list(cost_data['monthly_costs'].keys()),
                        'Cost (₹ Crores)': list(cost_data['monthly_costs'].values())
                    })
                    st.dataframe(monthly_df, use_container_width=True)
        
        with tab2:
            st.header("📅 Procurement Schedule Analysis")
            
            if isinstance(procurement_data, dict) and 'monthly_schedule' in procurement_data:
                schedule_df = pd.DataFrame(procurement_data['monthly_schedule'])
                
                # Material selector
                st.markdown("### 📊 Visualization Controls")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    material_list = ['All Materials'] + sorted(schedule_df['material'].tolist())
                    selected_material = st.selectbox(
                        "Select Material to View",
                        options=material_list,
                        key='procurement_material_selector'
                    )
                
                with col2:
                    chart_type = st.selectbox(
                        "Chart Type",
                        options=['Line Chart', 'Stacked Area', 'Heatmap'],
                        key='procurement_chart_type'
                    )
                
                st.markdown("---")
                
                # Display selected chart
                if chart_type == 'Line Chart':
                    material_filter = None if selected_material == 'All Materials' else selected_material
                    fig = create_procurement_timeline_chart(procurement_data, material_filter)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == 'Stacked Area':
                    fig = create_procurement_stacked_area_chart(procurement_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == 'Heatmap':
                    fig = create_material_heatmap(procurement_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Summary statistics
                st.markdown("### 📈 Procurement Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_materials = len(schedule_df)
                    st.metric("Total Materials", total_materials)
                
                with col2:
                    total_months = len(set([month for item in procurement_data['monthly_schedule'] 
                                           for month in item.get('monthly_distribution', {}).keys()]))
                    st.metric("Timeline Span", f"{total_months} months")
                
                with col3:
                    avg_lead_time = schedule_df['lead_time_months'].mean() if 'lead_time_months' in schedule_df else 0
                    st.metric("Avg Lead Time", f"{avg_lead_time:.1f} months")
                
                with col4:
                    total_qty = sum([item.get('total_quantity', 0) for item in procurement_data['monthly_schedule']])
                    st.metric("Total Quantity", f"{total_qty:.0f}")
                
                st.markdown("---")
                
                # Detailed material breakdown (collapsible)
                with st.expander("📦 Detailed Material Breakdown", expanded=False):
                    for idx, row in schedule_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"#### {row.get('material', 'Unknown')}")
                                st.write(f"**Category:** {row.get('category', 'N/A')}")
                                st.write(f"**Total Quantity:** {row.get('total_quantity', 0)} {row.get('unit', '')}")
                            
                            with col2:
                                st.write(f"**Peak Month:** {row.get('peak_month', 'N/A')}")
                                st.write(f"**Lead Time:** {row.get('lead_time_months', 'N/A')} months")
                            
                            if 'monthly_distribution' in row and isinstance(row['monthly_distribution'], dict):
                                monthly_dist_df = pd.DataFrame({
                                    'Month': list(row['monthly_distribution'].keys()),
                                    'Quantity': list(row['monthly_distribution'].values())
                                })
                                st.dataframe(monthly_dist_df, use_container_width=True, height=150)
                            
                            st.markdown("---")
                
                # Download procurement schedule
                st.markdown("### 📥 Export Data")
                csv = schedule_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Procurement Schedule (.csv)",
                    data=csv,
                    file_name=f"Procurement_Schedule_{pi['project_name'].replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No procurement schedule data available")
        
        with tab3:
            st.header("📋 Bill of Materials")
            st.dataframe(bom_df, use_container_width=True)
            
            csv = bom_df.to_csv(index=False)
            st.download_button(
                label="📥 Download BoM (.csv)",
                data=csv,
                file_name=f"BoM_{pi['project_name'].replace(' ', '_')}.csv",
                mime="text/csv")
        
        with tab4:
            st.header("📊 BoM Comparison")
            
            if current_project.get('comparison_df') is not None:
                comp_df = current_project['comparison_df']
                
                def highlight_status(row):
                    if row['Status'] == '✅ Match':
                        return ['background-color: #90EE90'] * len(row)
                    elif row['Status'] == '⚠️ Deviation':
                        return ['background-color: #FFD700'] * len(row)
                    elif row['Status'] == '🔴 Major Deviation':
                        return ['background-color: #FFB6C6'] * len(row)
                    elif row['Status'] == '❌ Missing':
                        return ['background-color: #FF6B6B'] * len(row)
                    elif row['Status'] == '➕ Additional':
                        return ['background-color: #87CEEB'] * len(row)
                    return [''] * len(row)
                
                styled_comp = comp_df.style.apply(highlight_status, axis=1)
                st.dataframe(styled_comp, use_container_width=True)
                
                # Download comparison
                comp_csv = comp_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison (.csv)",
                    data=comp_csv,
                    file_name=f"Comparison_{pi['project_name'].replace(' ', '_')}.csv",
                    mime="text/csv")
            else:
                st.info("No uploaded BoM to compare against.")
        
        with tab5:
            st.header("📄 Complete Report")
            
            full_report = f"""# POWERGRID Project Report

## Project: {pi['project_name']}

### Project Details
- Type: {pi['project_type']}
- Location: {pi['state']}, {pi['region']}
- Terrain: {pi['terrain']}
- Budget: ₹{pi['total_budget']} Crores
- Duration: {pi['procurement_duration']} months

### Cost Summary
- Total Material Cost: ₹{cost_data.get('total_cost', 0):.2f} Crores
- Budget Utilization: {budget_utilization:.1f}%
- Total Materials: {len(bom_df)}

### Monthly Cost Breakdown
"""
            
            if 'monthly_costs' in cost_data and cost_data['monthly_costs']:
                for month, cost in cost_data['monthly_costs'].items():
                    full_report += f"- {month}: ₹{cost:.2f} Cr\n"
            
            full_report += f"""

### Procurement Schedule Summary
- Total Materials: {len(procurement_data.get('monthly_schedule', []))}
- Analysis Date: {cost_data.get('analysis_date', 'N/A')}
"""
            
            st.markdown(full_report)
            
            st.download_button(
                label="📥 Download Full Report (.md)",
                data=full_report,
                file_name=f"Report_{pi['project_name'].replace(' ', '_')}.md",
                mime="text/markdown")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Project", use_container_width=True):
                current_project['status'] = 'configured'
                current_project['bom_df'] = None
                current_project['cost_data'] = None
                current_project['procurement_data'] = None
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Project", type="secondary", use_container_width=True):
                del st.session_state.projects[st.session_state.current_project_id]
                st.session_state.current_project_id = None
                st.rerun()

# --- Process All Projects Button ---
if len(st.session_state.projects) > 1:
    st.markdown("---")
    st.subheader("🚀 Batch Processing")
    
    uncompleted = [pid for pid, pdata in st.session_state.projects.items() 
                   if pdata['status'] != 'complete']
    
    if len(uncompleted) > 0:
        st.info(f"📊 {len(uncompleted)} project(s) pending completion")
        
        if st.button("⚡ Process All Pending Projects in Parallel", type="primary", use_container_width=True):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time
            
            progress_container = st.container()
            
            with progress_container:
                overall_progress = st.progress(0)
                status_text = st.empty()
                completed_count = 0
                
                def process_single_project(project_id, project_data):
                    """Process a single project (BoM + Costs + Procurement)"""
                    try:
                        project_name = project_data['project_inputs']['project_name']
                        
                        # Make a copy to avoid session state issues
                        project = {
                            'project_inputs': project_data['project_inputs'].copy(),
                            'status': project_data['status'],
                            'bom_df': project_data.get('bom_df'),
                            'cost_data': project_data.get('cost_data'),
                            'procurement_data': project_data.get('procurement_data'),
                            'uploaded_bom': project_data.get('uploaded_bom'),
                            'comparison_df': project_data.get('comparison_df'),
                            'created_at': project_data.get('created_at')
                        }
                        
                        # Generate BoM if needed
                        if project['status'] == 'configured':
                            material_prompts = create_all_material_bom_prompts(project['project_inputs'])
                            results = process_parallel_bom_generation(material_prompts, model, max_workers=20)
                            
                            bom_data = []
                            for r in results:
                                if r['success']:
                                    data = r['data']
                                    bom_data.append({
                                        'Material_Name': data.get('material_name', ''),
                                        'Unit': data.get('unit', ''),
                                        'Estimated_Quantity': float(data.get('estimated_quantity', 0)),
                                        'Priority': data.get('priority', 'Medium'),
                                        'Calculation_Method': data.get('calculation_method', ''),
                                        'Notes': data.get('notes', '')
                                    })
                            
                            project['bom_df'] = pd.DataFrame(bom_data)
                            project['status'] = 'bom_generated'
                        
                        # Calculate costs and procurement
                        if project['status'] == 'bom_generated' and project['bom_df'] is not None:
                            cost_data = calculate_static_costs(project['bom_df'], project['project_inputs']['region'])
                            project['cost_data'] = cost_data
                            
                            procurement_prompts = create_all_procurement_prompts(
                                project['bom_df'], project['project_inputs'])
                            procurement_json = process_parallel_procurement_generation(
                                procurement_prompts, model, max_workers=20)
                            
                            project['procurement_data'] = procurement_json
                            
                            monthly_costs = calculate_monthly_costs_from_procurement(
                                procurement_json, cost_data['material_costs'])
                            project['cost_data']['monthly_costs'] = monthly_costs
                            
                            project['status'] = 'complete'
                        
                        return {
                            'success': True, 
                            'project_id': project_id, 
                            'project_name': project_name,
                            'project_data': project  # Return the updated project data
                        }
                    
                    except Exception as e:
                        import traceback
                        return {
                            'success': False, 
                            'project_id': project_id, 
                            'project_name': project_data['project_inputs']['project_name'],
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }
                
                # Process all projects in parallel
                with ThreadPoolExecutor(max_workers=min(len(uncompleted), 5)) as executor:
                    # Submit all projects for processing with their data
                    future_to_project = {
                        executor.submit(process_single_project, pid, st.session_state.projects[pid]): pid 
                        for pid in uncompleted
                    }
                    
                    # Collect results as they complete
                    results = []
                    for future in as_completed(future_to_project):
                        result = future.result()
                        results.append(result)
                        completed_count += 1
                        
                        if result['success']:
                            status_text.text(f"✅ Completed: {result['project_name']} ({completed_count}/{len(uncompleted)})")
                        else:
                            status_text.text(f"❌ Failed: {result['project_name']} - {result.get('error', 'Unknown error')[:100]}")
                        
                        overall_progress.progress(completed_count / len(uncompleted))
                
                # Update session state with results (in main thread)
                success_count = 0
                fail_count = 0
                for result in results:
                    if result['success']:
                        st.session_state.projects[result['project_id']] = result['project_data']
                        success_count += 1
                    else:
                        fail_count += 1
                        # Show error details
                        with st.expander(f"❌ Error details for {result['project_name']}"):
                            st.error(result.get('error', 'Unknown error'))
                            if 'traceback' in result:
                                st.code(result['traceback'])
                
                overall_progress.empty()
                status_text.empty()
                
                # Show summary
                if success_count > 0:
                    st.success(f"🎉 Successfully completed {success_count} project(s)!")
                    st.balloons()
                
                if fail_count > 0:
                    st.warning(f"⚠️ {fail_count} project(s) failed. Check error details above.")
                
                # Small delay before rerun to show completion message
                time.sleep(2)
                st.rerun()

# --- Multi-Project Vendor Optimization Section ---
st.markdown("---")
st.header("🚚 Multi-Project Vendor & Logistics Optimization")

completed_projects = {pid: pdata for pid, pdata in st.session_state.projects.items() 
                     if pdata.get('status') == 'complete'}

if len(completed_projects) < 1:
    st.info("⏳ Complete at least 1 project to enable vendor optimization.")
elif len(completed_projects) == 1:
    st.info("💡 Vendor optimization works best with 2+ projects, but you can run it with 1 project.")
else:
    st.success(f"✅ {len(completed_projects)} completed projects available for optimization")

if len(completed_projects) > 0:
    with st.expander("🎯 About Vendor Optimization", expanded=False):
        st.markdown("""
        **What this does:**
        - Consolidates material demands across all completed projects
        - Intelligently pre-filters vendors and warehouses to reduce costs
        - Uses AI to determine optimal vendor selection strategy
        - Recommends logistics routes and warehouse usage
        - Provides cost analysis and risk assessment
        
        **Benefits of Multi-Project Optimization:**
        - Bulk ordering discounts (10-30% savings)
        - Reduced transport costs through route optimization
        - Risk diversification through vendor selection
        - Warehouse consolidation for regional projects
        """)
    
    # Show project selection
    st.markdown("### 📋 Projects to Include")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_projects = st.multiselect(
            "Select projects for vendor optimization",
            options=list(completed_projects.keys()),
            default=list(completed_projects.keys()),
            format_func=lambda x: f"{completed_projects[x]['project_inputs']['project_name']} ({completed_projects[x]['project_inputs']['state']})"
        )
    
    with col2:
        st.metric("Total Projects", len(selected_projects))
        if len(selected_projects) > 0:
            total_budget = sum(completed_projects[pid]['project_inputs']['total_budget'] 
                             for pid in selected_projects)
            st.metric("Combined Budget", f"₹{total_budget:.0f} Cr")
    
    if len(selected_projects) > 0:
        # Show material consolidation preview
        with st.expander("📊 Material Consolidation Preview", expanded=False):
            try:
                from multi_project_vendor_optimizer import MultiProjectVendorOptimizer
                
                # Create temporary optimizer for preview
                selected_project_dict = {pid: completed_projects[pid] for pid in selected_projects}
                temp_optimizer = MultiProjectVendorOptimizer(
                    selected_project_dict,
                    'Datasets/vendors.csv',
                    'Datasets/warehouse.csv',
                    model
                )
                
                consolidated = temp_optimizer.consolidate_demands_by_material()
                
                if consolidated:
                    summary_data = []
                    for category, data in consolidated.items():
                        summary_data.append({
                            'Material Category': category,
                            'Total Quantity (MT)': f"{data['total_quantity_mt']:.2f}",
                            'Projects': len(data['breakdown_by_project']),
                            'Geographic Spread': data.get('geographic_clustering', 'N/A'),
                            'Temporal Overlap': data.get('temporal_overlap', 'N/A')
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                else:
                    st.info("No material data available for consolidation")
                    
            except Exception as e:
                st.warning(f"Preview unavailable: {str(e)}")
        
        # Run optimization button
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Run Vendor & Logistics Optimization", type="primary", use_container_width=True):
                try:
                    from multi_project_vendor_optimizer import MultiProjectVendorOptimizer
                    
                    with st.spinner("🔍 Initializing optimizer..."):
                        selected_project_dict = {pid: completed_projects[pid] for pid in selected_projects}
                        optimizer = MultiProjectVendorOptimizer(
                            selected_project_dict,
                            'Datasets/vendors.csv',
                            'Datasets/warehouse.csv',
                            model
                        )
                    
                    with st.spinner("📊 Consolidating demands from all projects..."):
                        consolidated_demands = optimizer.consolidate_demands_by_material()
                        st.success(f"✅ Consolidated {len(consolidated_demands)} material categories")
                    
                    optimization_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    material_count = 0
                    for material_category, demand_data in consolidated_demands.items():
                        material_count += 1
                        status_text.text(f"🔄 Optimizing {material_category}... ({material_count}/{len(consolidated_demands)})")
                        
                        try:
                            with st.spinner(f"Pre-filtering vendors for {material_category}..."):
                                filtered_vendors = optimizer.filter_vendors_for_material(
                                    material_category, demand_data
                                )
                                
                                if len(filtered_vendors) == 0:
                                    optimization_results[material_category] = {
                                        'success': False,
                                        'error': 'No suitable vendors found'
                                    }
                                    continue
                                
                                filtered_warehouses = optimizer.filter_warehouses_for_projects(
                                    material_category, demand_data
                                )
                            
                            with st.spinner(f"AI analyzing optimal strategy for {material_category}..."):
                                result = optimizer.optimize_material_procurement(
                                    material_category, demand_data, 
                                    filtered_vendors, filtered_warehouses
                                )
                                optimization_results[material_category] = result
                        
                        except Exception as e:
                            optimization_results[material_category] = {
                                'success': False,
                                'error': str(e)
                            }
                        
                        progress_bar.progress(material_count / len(consolidated_demands))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results in session state
                    st.session_state['vendor_optimization_results'] = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'selected_projects': selected_projects,
                        'project_names': [completed_projects[pid]['project_inputs']['project_name'] 
                                        for pid in selected_projects],
                        'consolidated_demands': consolidated_demands,
                        'optimization_results': optimization_results
                    }
                    
                    # Count successes and failures
                    success_count = sum(1 for r in optimization_results.values() if r.get('success', False))
                    fail_count = len(optimization_results) - success_count
                    
                    if success_count > 0:
                        st.success(f"🎉 Successfully optimized {success_count} material categories!")
                        if fail_count > 0:
                            st.warning(f"⚠️ {fail_count} material(s) failed optimization")
                        st.balloons()
                    else:
                        st.error("❌ Optimization failed for all materials")
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

        with col2:
            if 'vendor_optimization_results' in st.session_state:
                if st.button("🔄 Clear Results", use_container_width=True):
                    del st.session_state['vendor_optimization_results']
                    st.rerun()
        
        with col3:
            pass  # Reserved for future actions

# --- Display Optimization Results ---
if 'vendor_optimization_results' in st.session_state:
    st.markdown("---")
    st.header("📊 Vendor Optimization Results")
    
    results_data = st.session_state['vendor_optimization_results']
    opt_results = results_data['optimization_results']
    consolidated_demands = results_data['consolidated_demands']
    
    # Summary metrics
    st.markdown(f"**Analysis Date:** {results_data['timestamp']}")
    st.markdown(f"**Projects Analyzed:** {', '.join(results_data['project_names'])}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_materials = len(opt_results)
        st.metric("Materials Analyzed", total_materials)
    
    with col2:
        success_count = sum(1 for r in opt_results.values() if r.get('success', False))
        st.metric("Successful Optimizations", success_count)
    
    with col3:
        total_vendors = 0
        for r in opt_results.values():
            if r.get('success') and r.get('result', {}).get('recommended_vendors'):
                total_vendors += len(r['result']['recommended_vendors'])
        st.metric("Total Vendors Selected", total_vendors)
    
    with col4:
        total_cost = 0
        for r in opt_results.values():
            if r.get('success') and r.get('result', {}).get('cost_summary', {}).get('total_procurement_cost_inr'):
                total_cost += r['result']['cost_summary']['total_procurement_cost_inr']
        st.metric("Total Procurement Cost", f"₹{total_cost/10000000:.2f} Cr")
    
    st.markdown("---")
    
    # Detailed results by material
    st.subheader("📦 Material-wise Optimization Details")
    
    for material_category in sorted(opt_results.keys()):
        result = opt_results[material_category]
        demand_data = consolidated_demands.get(material_category, {})
        
        with st.expander(f"{'✅' if result.get('success') else '❌'} {material_category} - {demand_data.get('total_quantity_mt', 0):.2f} MT", 
                        expanded=False):
            
            if not result.get('success'):
                st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
                continue
            
            opt_data = result.get('result', {})
            
            # Tabs for different aspects
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Strategy", "🏭 Vendors", "🚚 Logistics", "💰 Cost Analysis", "⚠️ Risk"
            ])
            
            with tab1:
                st.markdown("### Procurement Strategy")
                strategy = opt_data.get('strategy', 'UNKNOWN')
                st.info(f"**Strategy:** {strategy}")
                
                if 'strategy_reasoning' in opt_data:
                    st.markdown("**Reasoning:**")
                    st.write(opt_data['strategy_reasoning'])
                
                # Project breakdown
                st.markdown("### Project Requirements")
                project_breakdown = []
                for proj in demand_data.get('breakdown_by_project', []):
                    project_breakdown.append({
                        'Project': proj['project_name'],
                        'State': proj['state'],
                        'Quantity (MT)': f"{proj['quantity_needed_mt']:.2f}",
                        'Urgency': proj['urgency'],
                        'Delivery Months': ', '.join(proj['delivery_months'][:3])
                    })
                st.dataframe(pd.DataFrame(project_breakdown), use_container_width=True)
            
            with tab2:
                st.markdown("### Recommended Vendors")
                
                vendors = opt_data.get('recommended_vendors', [])
                if vendors:
                    vendor_data = []
                    for v in vendors:
                        vendor_data.append({
                            'Vendor ID': v.get('vendor_id', 'N/A'),
                            'Allocated Qty (MT)': f"{v.get('allocated_quantity_mt', 0):.2f}",
                            'Projects Served': ', '.join(v.get('serving_projects', [])),
                            'Reasoning': v.get('reasoning', 'N/A')[:100] + '...'
                        })
                    st.dataframe(pd.DataFrame(vendor_data), use_container_width=True)
                    
                    # Show full reasoning in expandable sections
                    for v in vendors:
                        with st.expander(f"Details: {v.get('vendor_id', 'N/A')}"):
                            st.markdown(f"**Full Reasoning:** {v.get('reasoning', 'N/A')}")
                            st.markdown(f"**Allocated Quantity:** {v.get('allocated_quantity_mt', 0):.2f} MT")
                            st.markdown(f"**Projects:** {', '.join(v.get('serving_projects', []))}")
                else:
                    st.info("No vendor recommendations available")
            
            with tab3:
                st.markdown("### Logistics Plan")
                
                logistics = opt_data.get('logistics_plan', {})
                routes = logistics.get('routes', [])
                
                if routes:
                    for i, route in enumerate(routes):
                        st.markdown(f"**Route {i+1}:**")
                        
                        route_legs = route.get('route_legs', [])
                        if route_legs:
                            leg_data = []
                            for leg in route_legs:
                                leg_data.append({
                                    'From': leg.get('from_location', 'N/A'),
                                    'To': leg.get('to_location', 'N/A'),
                                    'Mode': leg.get('transport_mode', 'N/A'),
                                    'Distance (km)': leg.get('distance_km', 0),
                                    'Cost (₹)': f"{leg.get('cost_inr', 0):,.0f}",
                                    'Duration (days)': leg.get('estimated_days', 0)
                                })
                            st.dataframe(pd.DataFrame(leg_data), use_container_width=True)
                        
                        if 'total_route_cost_inr' in route:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Route Cost", f"₹{route['total_route_cost_inr']/100000:.2f} L")
                            with col2:
                                st.metric("Route Duration", f"{route.get('total_duration_days', 0)} days")
                        
                        st.markdown("---")
                else:
                    st.info("No logistics routes available")
                
                # Warehouse decisions
                warehouse_decisions = opt_data.get('warehouse_decisions', {})
                if warehouse_decisions:
                    st.markdown("### Warehouse Usage")
                    wh_recommendation = warehouse_decisions.get('recommendation', 'N/A')
                    st.info(f"**Recommendation:** {wh_recommendation}")
                    
                    if warehouse_decisions.get('reasoning'):
                        st.write(warehouse_decisions['reasoning'])
            
            with tab4:
                st.markdown("### Cost Breakdown")
                
                cost_summary = opt_data.get('cost_summary', {})
                if cost_summary:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        try:
                            material_cost = float(cost_summary.get('material_cost_inr', 0))
                            st.metric("Material Cost", f"₹{material_cost/10000000:.2f} Cr")
                        except (ValueError, TypeError):
                            st.metric("Material Cost", "N/A")
                    
                    with col2:
                        try:
                            transport_cost = float(cost_summary.get('transport_cost_inr', 0))
                            st.metric("Transport Cost", f"₹{transport_cost/100000:.2f} L")
                        except (ValueError, TypeError):
                            st.metric("Transport Cost", "N/A")
                    
                    with col3:
                        try:
                            total_cost = float(cost_summary.get('total_procurement_cost_inr', 0))
                            st.metric("Total Cost", f"₹{total_cost/10000000:.2f} Cr")
                        except (ValueError, TypeError):
                            st.metric("Total Cost", "N/A")
                    
                    # Additional cost details
                    if 'gst_amount_inr' in cost_summary:
                        col1, col2 = st.columns(2)
                        with col1:
                            try:
                                gst = float(cost_summary['gst_amount_inr'])
                                st.metric("GST", f"₹{gst/100000:.2f} L")
                            except (ValueError, TypeError):
                                st.metric("GST", "N/A")
                        with col2:
                            if 'warehouse_cost_inr' in cost_summary:
                                try:
                                    wh_cost = float(cost_summary['warehouse_cost_inr'])
                                    st.metric("Warehouse", f"₹{wh_cost/100000:.2f} L")
                                except (ValueError, TypeError):
                                    st.metric("Warehouse", "N/A")
                    
                    # Savings
                    if 'estimated_savings_vs_individual_inr' in cost_summary:
                        try:
                            savings = float(cost_summary['estimated_savings_vs_individual_inr'])
                            st.success(f"💰 **Estimated Savings:** ₹{savings/100000:.2f} L (vs individual procurement)")
                        except (ValueError, TypeError):
                            st.info(f"💰 **Estimated Savings:** {cost_summary['estimated_savings_vs_individual_inr']}")
                else:
                    st.info("No cost summary available")
            
            with tab5:
                st.markdown("### Risk Assessment")
                
                risk_assessment = opt_data.get('risk_assessment', {})
                if risk_assessment:
                    risk_level = risk_assessment.get('overall_risk_level', 'UNKNOWN')
                    
                    risk_color = {
                        'LOW': '🟢',
                        'MEDIUM': '🟡',
                        'HIGH': '🔴'
                    }.get(risk_level, '⚪')
                    
                    st.markdown(f"**Overall Risk Level:** {risk_color} {risk_level}")
                    
                    # Key risks
                    key_risks = risk_assessment.get('key_risks', [])
                    if key_risks:
                        st.markdown("**Key Risks:**")
                        for risk in key_risks:
                            st.warning(f"⚠️ {risk}")
                    
                    # Mitigation strategies
                    mitigations = risk_assessment.get('mitigation_strategies', [])
                    if mitigations:
                        st.markdown("**Mitigation Strategies:**")
                        for mitigation in mitigations:
                            st.info(f"✓ {mitigation}")
                else:
                    st.info("No risk assessment available")
                
                # Timeline feasibility
                timeline = opt_data.get('timeline_feasibility', {})
                if timeline:
                    st.markdown("### Timeline Feasibility")
                    
                    feasible = timeline.get('feasible', False)
                    st.markdown(f"**Feasible:** {'✅ Yes' if feasible else '❌ No'}")
                    
                    if 'reasoning' in timeline:
                        st.write(timeline['reasoning'])
                    
                    if 'estimated_procurement_duration_days' in timeline:
                        st.metric("Estimated Duration", f"{timeline['estimated_procurement_duration_days']} days")
    
    # Download complete report
    st.markdown("---")
    st.subheader("📄 Export Results")
    
    if st.button("📥 Generate Complete Optimization Report", use_container_width=True):
        report = f"""# Multi-Project Vendor Optimization Report

**Generated:** {results_data['timestamp']}

**Projects Analyzed:**
{chr(10).join(f"- {name}" for name in results_data['project_names'])}

---

## Executive Summary

- **Total Materials:** {len(opt_results)}
- **Successful Optimizations:** {success_count}
- **Total Vendors Selected:** {total_vendors}
- **Total Procurement Cost:** ₹{total_cost/10000000:.2f} Crores

---

## Material-wise Optimization Details

"""
        
        for material_category in sorted(opt_results.keys()):
            result = opt_results[material_category]
            demand_data = consolidated_demands.get(material_category, {})
            
            report += f"""
### {material_category}

**Status:** {'✅ Success' if result.get('success') else '❌ Failed'}
**Total Demand:** {demand_data.get('total_quantity_mt', 0):.2f} MT
**Projects:** {len(demand_data.get('breakdown_by_project', []))}

"""
            
            if result.get('success'):
                opt_data = result['result']
                
                report += f"""**Strategy:** {opt_data.get('strategy', 'N/A')}

**Vendors:**
"""
                for v in opt_data.get('recommended_vendors', []):
                    report += f"""- {v.get('vendor_id', 'N/A')}: {v.get('allocated_quantity_mt', 0):.2f} MT
  Projects: {', '.join(v.get('serving_projects', []))}
"""
                
                cost_summary = opt_data.get('cost_summary', {})
                if cost_summary:
                    report += f"""
**Cost Summary:**
- Material Cost: ₹{cost_summary.get('material_cost_inr', 0)/10000000:.2f} Cr
- Transport Cost: ₹{cost_summary.get('transport_cost_inr', 0)/100000:.2f} L
- Total Cost: ₹{cost_summary.get('total_procurement_cost_inr', 0)/10000000:.2f} Cr
"""

        # Download button for the report
        st.download_button(
            label="📥 Download Optimization Report (.md)",
            data=report,
            file_name="optimization_report.md",
            mime="text/markdown"
        )
