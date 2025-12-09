# multi_project_vendor_optimizer.py

"""
Multi-Project Vendor & Logistics Optimization System
Uses intelligent pre-filtering + LLM reasoning for vendor selection and route optimization
"""

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# State coordinates for distance calculation
STATE_COORDINATES = {
    'Gujarat': (22.2587, 71.1924),
    'Maharashtra': (19.7515, 75.7139),
    'Karnataka': (15.3173, 75.7139),
    'Tamil Nadu': (11.1271, 78.6569),
    'Rajasthan': (27.5922, 75.5885),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Haryana': (29.0588, 77.0745),
    'Odisha': (20.9517, 85.0985),
    'West Bengal': (24.3745, 88.4790),
    'Jharkhand': (23.6102, 85.2799),
    'Punjab': (31.1471, 75.3412),
    'Delhi': (28.7041, 77.1025),
    'Himachal Pradesh': (31.7433, 77.1205),
    'Uttarakhand': (30.0668, 79.0193),
    'Bihar': (25.0961, 85.3131),
    'Andhra Pradesh': (15.9129, 79.7400),
    'Telangana': (18.1124, 79.0193),
    'Kerala': (10.8505, 76.2711),
    'Chhattisgarh': (21.2787, 81.8661),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Goa': (15.2993, 74.1240),
    'Assam': (26.2006, 92.9376),
    'Sikkim': (27.5330, 88.5122),
    'Meghalaya': (25.4670, 91.3662),
    'Manipur': (24.6637, 93.9063),
    'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624),
    'Tripura': (23.9408, 91.9882),
}

# Regional adjacency for proximity scoring
REGION_ADJACENCY = {
    'Northern': ['Northern', 'Western', 'Eastern'],
    'Western': ['Western', 'Northern', 'Southern'],
    'Southern': ['Southern', 'Western', 'Eastern'],
    'Eastern': ['Eastern', 'Northern', 'Southern'],
    'North-Eastern': ['North-Eastern', 'Eastern'],
}

# Material category mapping (BoM material name → Vendor category)
MATERIAL_CATEGORY_MAPPING = {
    'steel': 'Steel',
    'galvanized': 'Steel',
    'lattice': 'Steel',
    'reinforcement': 'Steel',
    'conductor': 'Conductor',
    'acsr': 'Conductor',
    'cable': 'Conductor',
    'wire': 'Conductor',
    'earth': 'Conductor',
    'insulator': 'Insulator',
    'disc': 'Insulator',
    'suspension': 'Insulator',
    'tension': 'Insulator',
    'concrete': 'Concrete',
    'cement': 'Concrete',
    'foundation': 'Concrete',
    'hardware': 'Hardware',
    'bolt': 'Hardware',
    'nut': 'Hardware',
    'clamp': 'Hardware',
    'socket': 'Hardware',
    'guard': 'Hardware',
    'plate': 'Hardware',
    'marker': 'Hardware',
    'earthing': 'Hardware',
    'grounding': 'Hardware',
}

# Transport cost matrix
TRANSPORT_COSTS = {
    'road': {
        'cost_per_mt_per_km': 6.0,
        'speed_km_per_day': 400,
        'best_for': 'distances < 500 km or last-mile delivery',
        'loading_time_days': 0.5
    },
    'rail': {
        'cost_per_mt_per_km': 2.3,
        'speed_km_per_day': 250,
        'best_for': 'distances > 500 km, bulk orders > 100 MT',
        'loading_unloading_time_days': 3,
        'major_terminals': ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Jamshedpur', 'Bengaluru']
    },
    'coastal_ship': {
        'cost_per_mt_per_km_equivalent': 1.2,
        'speed_km_per_day': 150,
        'best_for': 'coastal routes, very large orders > 500 MT',
        'loading_unloading_time_days': 5,
        'applicable_states': ['Gujarat', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Odisha', 'Kerala', 'Goa']
    }
}

# Business rules for optimization
BUSINESS_RULES = {
    'max_vendors_per_material': 3,
    'min_order_quantity_per_vendor_mt': 50,
    'budget_buffer_percent': 10,
    'risk_diversification_threshold': 0.7,
    'warehouse_cost_threshold': 1.15,
    'bulk_discount_min_qty': 500,
}

# Optimization priorities (weights)
OPTIMIZATION_WEIGHTS = {
    'cost': 50,
    'delivery_time': 25,
    'reliability': 15,
    'risk_diversification': 10
}


class MultiProjectVendorOptimizer:
    """
    Optimizes vendor selection and logistics for multiple projects simultaneously.
    Uses intelligent pre-filtering + LLM reasoning.
    """
    
    def __init__(self, projects_dict: Dict, vendors_csv_path: str, warehouses_csv_path: str, gemini_model):
        """
        Initialize optimizer with projects data and resource CSVs.
        
        Args:
            projects_dict: st.session_state.projects dictionary
            vendors_csv_path: Path to vendors.csv
            warehouses_csv_path: Path to warehouse.csv
            gemini_model: Configured Gemini model
        """
        self.projects = {pid: pdata for pid, pdata in projects_dict.items() 
                        if pdata.get('status') == 'complete'}
        self.vendors_df = pd.read_csv(vendors_csv_path)
        self.warehouses_df = pd.read_csv(warehouses_csv_path)
        self.model = gemini_model
        
        # Calculate available capacities for warehouses
        for material in ['steel', 'conductor', 'insulator', 'concrete', 'hardware']:
            cap_col = f'{material}_capacity_mt'
            stock_col = f'{material}_current_stock_mt'
            if cap_col in self.warehouses_df.columns and stock_col in self.warehouses_df.columns:
                self.warehouses_df[f'{material}_available_capacity_mt'] = (
                    self.warehouses_df[cap_col] - self.warehouses_df[stock_col]
                )
        
        self.consolidated_demands = {}
        self.optimization_results = {}
    
    def calculate_distance(self, state1: str, state2: str) -> float:
        """Calculate approximate distance between two states in km."""
        if state1 not in STATE_COORDINATES or state2 not in STATE_COORDINATES:
            return 500.0
        
        lat1, lon1 = STATE_COORDINATES[state1]
        lat2, lon2 = STATE_COORDINATES[state2]
        
        dlat = abs(lat1 - lat2) * 111
        dlon = abs(lon1 - lon2) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
        
        return np.sqrt(dlat**2 + dlon**2) * 1.3
    
    def map_material_to_category(self, material_name: str) -> str:
        """Map BoM material name to vendor category."""
        material_lower = material_name.lower()
        
        for keyword, category in MATERIAL_CATEGORY_MAPPING.items():
            if keyword in material_lower:
                return category
        
        return 'Hardware'
    
    def consolidate_demands_by_material(self):
        """Aggregate demands from all projects by material category."""
        material_demands = {}
        
        for project_id, project_data in self.projects.items():
            bom_df = project_data.get('bom_df')
            if bom_df is None or len(bom_df) == 0:
                continue
            
            project_inputs = project_data['project_inputs']
            procurement_data = project_data.get('procurement_data', {})
            
            for _, row in bom_df.iterrows():
                material_name = row['Material_Name']
                quantity = float(row['Estimated_Quantity'])
                unit = row['Unit']
                
                category = self.map_material_to_category(material_name)
                
                delivery_months = []
                if procurement_data and 'monthly_schedule' in procurement_data:
                    for mat_sched in procurement_data['monthly_schedule']:
                        if mat_sched.get('material', '').lower() in material_name.lower():
                            if 'monthly_distribution' in mat_sched:
                                delivery_months = list(mat_sched['monthly_distribution'].keys())
                            break
                
                if category not in material_demands:
                    material_demands[category] = {
                        'material_category': category,
                        'total_quantity_mt': 0,
                        'unit': unit,
                        'breakdown_by_project': []
                    }
                
                material_demands[category]['total_quantity_mt'] += quantity
                material_demands[category]['breakdown_by_project'].append({
                    'project_id': project_id,
                    'project_name': project_inputs['project_name'],
                    'state': project_inputs['state'],
                    'region': project_inputs['region'],
                    'quantity_needed_mt': quantity,
                    'delivery_months': delivery_months[:3] if delivery_months else ['Month_1'],
                    'urgency': row.get('Priority', 'Medium').upper()
                })
        
        for category, data in material_demands.items():
            projects_by_region = {}
            all_months = set()
            
            for proj in data['breakdown_by_project']:
                region = proj['region']
                if region not in projects_by_region:
                    projects_by_region[region] = []
                projects_by_region[region].append(proj['project_name'])
                
                all_months.update(proj['delivery_months'])
            
            data['geographic_clustering'] = ', '.join([
                f"{len(projs)} in {reg}" for reg, projs in projects_by_region.items()
            ])
            
            data['temporal_overlap'] = f"{len(all_months)} unique delivery months"
        
        self.consolidated_demands = material_demands
        return material_demands
    
    def filter_vendors_for_material(self, material_category: str, demand_data: Dict) -> pd.DataFrame:
        """Pre-filter vendors using deterministic logic before LLM."""
        total_qty = demand_data['total_quantity_mt']
        projects = demand_data['breakdown_by_project']
        project_states = [p['state'] for p in projects]
        project_regions = list(set([p['region'] for p in projects]))
        
        category_vendors = self.vendors_df[
            self.vendors_df['material_category'] == material_category
        ].copy()
        
        if len(category_vendors) == 0:
            return pd.DataFrame()
        
        min_capacity = total_qty * 0.2
        category_vendors['available_capacity_mt'] = category_vendors['capacity_mt_yr'] * 0.8
        
        capable_vendors = category_vendors[
            category_vendors['available_capacity_mt'] >= min_capacity
        ]
        
        if len(capable_vendors) < 3:
            capable_vendors = category_vendors
        
        capable_vendors['geo_score'] = 0.0
        
        for idx, row in capable_vendors.iterrows():
            vendor_state = row['state']
            vendor_region = self.get_region_for_state(vendor_state)
            
            if vendor_state in project_states:
                capable_vendors.loc[idx, 'geo_score'] = 20.0
            elif vendor_region in project_regions:
                capable_vendors.loc[idx, 'geo_score'] = 15.0
            elif vendor_region in [adj for r in project_regions for adj in REGION_ADJACENCY.get(r, [])]:
                capable_vendors.loc[idx, 'geo_score'] = 10.0
            else:
                capable_vendors.loc[idx, 'geo_score'] = 5.0
        
        min_price = capable_vendors['current_price_per_mt'].min()
        max_price = capable_vendors['current_price_per_mt'].max()
        
        capable_vendors['price_score'] = 30 * (1 - (capable_vendors['current_price_per_mt'] - min_price) / (max_price - min_price + 1))
        
        capable_vendors['capacity_score'] = capable_vendors['available_capacity_mt'].apply(
            lambda cap: 25 if total_qty * 0.5 <= cap <= total_qty * 2 else 
                       20 if cap >= total_qty * 2 else 
                       15 * (cap / (total_qty * 0.5))
        )
        
        min_lead = capable_vendors['lead_time_days'].min()
        max_lead = capable_vendors['lead_time_days'].max()
        capable_vendors['lead_time_score'] = 15 * (1 - (capable_vendors['lead_time_days'] - min_lead) / (max_lead - min_lead + 1))
        
        capable_vendors['discount_score'] = capable_vendors['bulk_discount_pct']
        
        capable_vendors['relevance_score'] = (
            capable_vendors['price_score'] +
            capable_vendors['capacity_score'] +
            capable_vendors['geo_score'] +
            capable_vendors['lead_time_score'] +
            capable_vendors['discount_score']
        )
        
        top_vendors = capable_vendors.nlargest(6, 'relevance_score')
        
        return top_vendors
    
    def filter_warehouses_for_projects(self, material_category: str, demand_data: Dict) -> pd.DataFrame:
        """Pre-filter warehouses before sending to LLM."""
        projects = demand_data['breakdown_by_project']
        project_regions = list(set([p['region'] for p in projects]))
        
        regional_warehouses = self.warehouses_df[
            self.warehouses_df['region'].isin(project_regions)
        ].copy()
        
        material_lower = material_category.lower()
        cap_col = f'{material_lower}_available_capacity_mt'
        storage_cost_col = f'storage_cost_{material_lower}_per_mt_month'
        
        if cap_col in regional_warehouses.columns:
            capable_warehouses = regional_warehouses[
                regional_warehouses[cap_col] >= 50
            ]
        else:
            capable_warehouses = regional_warehouses
        
        if len(capable_warehouses) < 3:
            capable_warehouses = self.warehouses_df[
                self.warehouses_df[cap_col] >= 50
            ].copy() if cap_col in self.warehouses_df.columns else self.warehouses_df.copy()
        
        if storage_cost_col in capable_warehouses.columns:
            median_cost = capable_warehouses[storage_cost_col].median()
            
            capable_warehouses['cost_score'] = capable_warehouses[storage_cost_col].apply(
                lambda cost: 50 if cost <= median_cost else 
                            25 if cost <= median_cost * 1.5 else 10
            )
        else:
            capable_warehouses['cost_score'] = 25
        
        capable_warehouses['region_score'] = capable_warehouses['region'].apply(
            lambda r: 50 if r in project_regions else 25
        )
        
        capable_warehouses['warehouse_score'] = (
            capable_warehouses['cost_score'] + capable_warehouses['region_score']
        )
        
        top_warehouses = capable_warehouses.nlargest(5, 'warehouse_score')
        
        return top_warehouses
    
    def get_region_for_state(self, state: str) -> str:
        """Get region for a given state."""
        REGIONS = {
            'Northern Region': ['Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Punjab', 'Rajasthan', 'Uttar Pradesh', 'Uttarakhand'],
            'Western Region': ['Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Chhattisgarh', 'Goa', 'Daman & Diu', 'Dadra & Nagar Haveli'],
            'Southern Region': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 'Puducherry'],
            'Eastern Region': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Sikkim'],
            'North-Eastern Region': ['Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Tripura']
        }
        
        for region, states in REGIONS.items():
            if state in states:
                return region
        return 'Unknown'
    
    def create_llm_prompt_for_material(self, material_category: str, demand_data: Dict, 
                                       filtered_vendors: pd.DataFrame, 
                                       filtered_warehouses: pd.DataFrame) -> str:
        """Create comprehensive LLM prompt for vendor selection and logistics optimization."""
        
        vendors_json = filtered_vendors[[
            'vendor_id', 'vendor_name', 'state', 'capacity_mt_yr', 'available_capacity_mt',
            'lead_time_days', 'current_price_per_mt', 'gst_pct', 'bulk_discount_pct'
        ]].to_dict(orient='records')
        
        for vendor in vendors_json:
            discount_pct = vendor['bulk_discount_pct']
            vendor['bulk_discount_tiers'] = [
                {'min_qty_mt': int(demand_data['total_quantity_mt'] * 0.3), 'discount_pct': discount_pct * 0.5},
                {'min_qty_mt': int(demand_data['total_quantity_mt'] * 0.6), 'discount_pct': discount_pct * 0.75},
                {'min_qty_mt': int(demand_data['total_quantity_mt']), 'discount_pct': discount_pct}
            ]
            vendor['location_coordinates'] = STATE_COORDINATES.get(vendor['state'], (0, 0))
        
        warehouses_json = []
        if len(filtered_warehouses) > 0:
            material_lower = material_category.lower()
            cap_col = f'{material_lower}_available_capacity_mt'
            cost_col = f'storage_cost_{material_lower}_per_mt_month'
            
            for _, wh in filtered_warehouses.iterrows():
                wh_data = {
                    'warehouse_id': wh['warehouse_id'],
                    'name': wh['warehouse_name'],
                    'state': wh['state'],
                    'region': wh['region'],
                }
                
                if cap_col in wh:
                    wh_data['available_capacity_mt'] = float(wh[cap_col])
                if cost_col in wh:
                    wh_data['storage_cost_per_mt_per_month'] = float(wh[cost_col])
                
                warehouses_json.append(wh_data)
        
        prompt = f"""You are an expert supply chain optimizer for POWERGRID India. Analyze multi-project material procurement and recommend optimal vendor selection and logistics strategy.

**TASK:**
Optimize {material_category} procurement for {len(demand_data['breakdown_by_project'])} project(s) with total demand {demand_data['total_quantity_mt']:.2f} MT.

**CONSOLIDATED DEMAND:**
{json.dumps(demand_data, indent=2)}

**AVAILABLE VENDORS:**
{json.dumps(vendors_json, indent=2)}

**AVAILABLE WAREHOUSES:**
{json.dumps(warehouses_json, indent=2)}

**TRANSPORT COSTS:**
- Road: ₹6/MT/km (< 500 km)
- Rail: ₹2.3/MT/km (> 500 km, bulk > 100 MT)
- Ship: ₹1.2/MT/km (coastal, bulk > 500 MT)

**RULES:**
- Max {BUSINESS_RULES['max_vendors_per_material']} vendors per material
- Min {BUSINESS_RULES['min_order_quantity_per_vendor_mt']} MT per vendor
- Avoid single vendor >70% allocation

**OUTPUT (JSON only, no markdown):**
{{
  "strategy": "CONSOLIDATED | SPLIT_BY_REGION | SPLIT_BY_PROJECT",
  "strategy_reasoning": "Brief explanation",
  "recommended_vendors": [
    {{
      "vendor_id": "V_XXX",
      "allocated_quantity_mt": 0,
      "serving_projects": ["Project A"],
      "reasoning": "Why selected"
    }}
  ],
  "logistics_plan": {{
    "routes": [{{
      "route_legs": [{{
        "from_location": "State",
        "to_location": "State",
        "transport_mode": "road|rail|coastal_ship",
        "distance_km": 0,
        "cost_inr": 0,
        "estimated_days": 0
      }}],
      "total_route_cost_inr": 0,
      "total_duration_days": 0
    }}]
  }},
  "warehouse_decisions": {{
    "recommendation": "USE_WAREHOUSES | DIRECT_DELIVERY | HYBRID",
    "selected_warehouses": [],
    "reasoning": "Explanation"
  }},
  "cost_summary": {{
    "material_cost_inr": 0,
    "transport_cost_inr": 0,
    "gst_amount_inr": 0,
    "warehouse_cost_inr": 0,
    "total_procurement_cost_inr": 0,
    "estimated_savings_vs_individual_inr": 0
  }},
  "risk_assessment": {{
    "overall_risk_level": "LOW | MEDIUM | HIGH",
    "key_risks": ["Risk 1"],
    "mitigation_strategies": ["Strategy 1"]
  }},
  "timeline_feasibility": {{
    "feasible": true,
    "estimated_procurement_duration_days": 0,
    "reasoning": "Timeline explanation"
  }}
}}

Return ONLY valid JSON, no additional text."""
        
        return prompt
    
    def parse_llm_response(self, response_text: str) -> dict:
        """Parse and validate LLM response."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Extract from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Find largest JSON object
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx + 1]
                return json.loads(json_str)
        except Exception:
            pass
        
        return None
    
    def optimize_material_procurement(self, material_category: str, demand_data: Dict,
                                     filtered_vendors: pd.DataFrame, 
                                     filtered_warehouses: pd.DataFrame) -> Dict:
        """Optimize procurement for a single material category using LLM reasoning."""
        try:
            prompt = self.create_llm_prompt_for_material(
                material_category, demand_data, filtered_vendors, filtered_warehouses
            )
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return {
                    'success': False,
                    'error': 'No response from LLM'
                }
            
            # Parse response
            result = self.parse_llm_response(response.text)
            
            if not result:
                return {
                    'success': False,
                    'error': 'Failed to parse LLM response',
                    'raw_response': response.text[:1000]
                }
            
            # Validate required fields
            required_fields = ['strategy', 'recommended_vendors', 'cost_summary']
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}',
                    'partial_result': result
                }
            
            return {
                'success': True,
                'result': result,
                'raw_response': response.text
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def optimize_all_materials(self):
        """Run optimization for all consolidated materials in parallel."""
        if not self.consolidated_demands:
            self.consolidate_demands_by_material()
        
        results = {}
        
        def optimize_single_material(material_category, demand_data):
            """Process a single material category."""
            try:
                filtered_vendors = self.filter_vendors_for_material(material_category, demand_data)
                filtered_warehouses = self.filter_warehouses_for_projects(material_category, demand_data)
                
                if len(filtered_vendors) == 0:
                    return (material_category, {
                        'success': False,
                        'error': 'No suitable vendors found'
                    })
                
                # Take top 3 vendors instead of 6 (50% reduction)
                filtered_vendors = filtered_vendors.nlargest(3, 'relevance_score')
                # Take top 2 warehouses instead of 5 (60% reduction)
                filtered_warehouses = filtered_warehouses.nlargest(2, 'warehouse_score') if len(filtered_warehouses) > 0 else filtered_warehouses
                
                result = self.optimize_material_procurement(
                    material_category, demand_data, filtered_vendors, filtered_warehouses
                )
                
                return (material_category, result)
            
            except Exception as e:
                import traceback
                return (material_category, {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        # Process all materials in parallel - FIX HERE
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all jobs at once
            future_to_material = {
                executor.submit(optimize_single_material, cat, data): cat
                for cat, data in self.consolidated_demands.items()
            }
            
            # Collect results as they complete (non-blocking)
            from concurrent.futures import as_completed
            for future in as_completed(future_to_material):
                material_category, result = future.result()
                results[material_category] = result
        
        self.optimization_results = results
        return results
    
    def generate_summary_report(self) -> str:
        """Generate text summary of optimization results."""
        if not self.optimization_results:
            return "No optimization results available."
        
        report = "# Multi-Project Vendor Optimization Summary\n\n"
        
        success_count = sum(1 for r in self.optimization_results.values() if r.get('success'))
        total_count = len(self.optimization_results)
        
        report += f"**Optimization Status:** {success_count}/{total_count} materials successfully optimized\n\n"
        
        for material_category, result in self.optimization_results.items():
            report += f"## {material_category}\n\n"
            
            if result.get('success'):
                opt_data = result['result']
                report += f"- **Strategy:** {opt_data.get('strategy', 'N/A')}\n"
                report += f"- **Vendors:** {len(opt_data.get('recommended_vendors', []))}\n"
                
                cost_summary = opt_data.get('cost_summary', {})
                if cost_summary:
                    total_cost = cost_summary.get('total_procurement_cost_inr', 0)
                    report += f"- **Total Cost:** ₹{total_cost/10000000:.2f} Crores\n"
                
                report += "\n"
            else:
                report += f"- **Error:** {result.get('error', 'Unknown error')}\n\n"
        
        return report


def run_multi_project_optimization(projects_dict: Dict, vendors_csv: str, 
                                   warehouses_csv: str, gemini_model) -> Dict:
    """
    Main entry point for multi-project vendor optimization.
    
    Args:
        projects_dict: st.session_state.projects (filtered for complete projects)
        vendors_csv: Path to vendors.csv
        warehouses_csv: Path to warehouse.csv
        gemini_model: Configured Gemini model
    
    Returns:
        dict with optimization results
    """
    optimizer = MultiProjectVendorOptimizer(
        projects_dict, vendors_csv, warehouses_csv, gemini_model
    )
    
    consolidated_demands = optimizer.consolidate_demands_by_material()
    optimization_results = optimizer.optimize_all_materials()
    
    return {
        'consolidated_demands': consolidated_demands,
        'optimization_results': optimization_results,
        'summary_report': optimizer.generate_summary_report()
    }