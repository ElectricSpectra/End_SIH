from knowledge_base import (BASELINE_MATERIALS_220KV, VOLTAGE_MULTIPLIERS,
                            TERRAIN_MULTIPLIERS, S_CURVE_PATTERNS,
                            EXAMPLE_CALCULATIONS, PROCUREMENT_BEST_PRACTICES)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable
import json
import re


def get_project_context(project_inputs: Dict) -> Dict:
    """Extract common project context used across all material prompts"""
    voltage = project_inputs['project_type'].split()[0]
    terrain = project_inputs['terrain']
    line_length = project_inputs.get('line_length', 100)
    scale_factor = line_length / 100

    voltage_key = f"{voltage}kV"
    voltage_mult = VOLTAGE_MULTIPLIERS.get(voltage_key,
                                           VOLTAGE_MULTIPLIERS["220kV"])
    terrain_mult = TERRAIN_MULTIPLIERS.get(terrain,
                                           TERRAIN_MULTIPLIERS["Plain"])

    return {
        'voltage': voltage,
        'voltage_key': voltage_key,
        'terrain': terrain,
        'line_length': line_length,
        'scale_factor': scale_factor,
        'voltage_mult': voltage_mult,
        'terrain_mult': terrain_mult,
        'project_inputs': project_inputs
    }


def create_single_material_bom_prompt(material_name: str, material_data: Dict,
                                      context: Dict) -> str:
    """Generate a prompt for a single material's BoM calculation"""

    voltage_mult = context['voltage_mult']
    terrain_mult = context['terrain_mult']
    scale_factor = context['scale_factor']
    project_inputs = context['project_inputs']

    category = material_data['category']
    if category == 'steel':
        v_mult = voltage_mult['steel_mult']
        t_mult = terrain_mult['steel_mult']
    elif category == 'concrete':
        v_mult = voltage_mult['concrete_mult']
        t_mult = terrain_mult['concrete_mult']
    elif category == 'insulator':
        v_mult = voltage_mult['insulators_mult']
        t_mult = 1.0
    elif category == 'conductor':
        v_mult = 1.0 if context['voltage_key'] == '220kV' else 1.1
        t_mult = 1.0
    else:
        v_mult = 1.0
        t_mult = 1.0

    prompt = f"""You are a POWERGRID Senior Engineer. Calculate the quantity for this single material.

**PROJECT:**
- Type: {project_inputs['project_type']}
- Voltage: {context['voltage_key']}
- Line Length: {context['line_length']} km
- Terrain: {context['terrain']}
- Location: {project_inputs['state']}, {project_inputs['region']}

**MATERIAL:** {material_name}
- Baseline Quantity (220kV, 100km, Plain): {material_data['qty_range']} {material_data['unit']}
- Category: {material_data['category']}
- Specs: {material_data['specs']}
- Priority: {material_data['priority']}

**MULTIPLIERS:**
- Scale Factor: {scale_factor}x
- Voltage Multiplier: {v_mult}x
- Terrain Multiplier: {t_mult}x

**CALCULATION:**
Final = Baseline_Avg × {scale_factor} × {v_mult} × {t_mult}

**OUTPUT FORMAT (JSON only, no markdown):**
{{"material_name": "{material_name}", "unit": "{material_data['unit']}", "estimated_quantity": <calculated_number>, "priority": "{material_data['priority']}", "calculation_method": "<formula_with_values>", "notes": "<engineering_reasoning>"}}

Output ONLY the JSON object, no other text."""

    return prompt


def create_all_material_bom_prompts(project_inputs: Dict) -> List[Dict]:
    """Create individual prompts for each material"""
    context = get_project_context(project_inputs)
    prompts = []

    for material_name, material_data in BASELINE_MATERIALS_220KV.items():
        prompt = create_single_material_bom_prompt(material_name,
                                                   material_data, context)
        prompts.append({
            'material_name': material_name,
            'material_data': material_data,
            'prompt': prompt
        })

    return prompts


def process_parallel_bom_generation(
        prompts: List[Dict],
        model,
        max_workers: int = 20,
        progress_callback: Callable = None) -> List[Dict]:
    """Process BoM generation prompts in parallel using ThreadPoolExecutor"""
    results = []
    total = len(prompts)

    def generate_for_material(prompt_data: Dict) -> Dict:
        try:
            response = model.generate_content(prompt_data['prompt'])
            response_text = response.text.strip()

            if response_text.startswith('```'):
                response_text = re.sub(r'^```json?\s*|\s*```$',
                                       '',
                                       response_text,
                                       flags=re.MULTILINE)

            result = json.loads(response_text)
            return {
                'success': True,
                'material_name': prompt_data['material_name'],
                'data': result
            }
        except Exception as e:
            return {
                'success': False,
                'material_name': prompt_data['material_name'],
                'error': str(e),
                'data': {
                    'material_name': prompt_data['material_name'],
                    'unit': prompt_data['material_data']['unit'],
                    'estimated_quantity': 0,
                    'priority': prompt_data['material_data']['priority'],
                    'calculation_method': 'Error in generation',
                    'notes': f'Error: {str(e)}'
                }
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(generate_for_material, p): p
            for p in prompts
        }

        for idx, future in enumerate(as_completed(future_to_prompt), 1):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback(idx, total, result['material_name'],
                                  result['success'])

    return results


def create_single_material_procurement_prompt(material_name: str,
                                              quantity: float, unit: str,
                                              category: str,
                                              project_inputs: Dict) -> str:
    """Generate a procurement prompt for a single material"""

    duration = project_inputs['procurement_duration']
    s_curve = S_CURVE_PATTERNS.get(category, S_CURVE_PATTERNS['hardware'])

    prompt = f"""You are a POWERGRID Procurement Strategist. Create a procurement schedule for this single material.

**MATERIAL:** {material_name}
- Total Quantity: {quantity} {unit}
- Category: {category}
- Project Duration: {duration} months
- Location: {project_inputs['state']}, {project_inputs['region']}
- Monsoon Factor: {"Enabled" if project_inputs.get('monsoon_factor', False) else "Disabled"}

**S-CURVE PATTERN FOR {category.upper()}:**
- Description: {s_curve['description']}
- Lead Time: {s_curve['lead_time']}
- Order Timing: {s_curve['order_timing']}
- Risk Level: {s_curve['risk_level']}

**Phases:**"""

    for phase, details in s_curve['phases'].items():
        prompt += f"\n- {phase}: {details['months']} months = {details['percentage']} ({details['breakdown']})"

    prompt += f"""

**OUTPUT FORMAT (JSON only, no markdown):**
{{
  "material": "{material_name}",
  "category": "{category}",
  "unit": "{unit}",
  "total_quantity": {quantity},
  "monthly_distribution": {{
    "month_1": <qty>,
    "month_2": <qty>,
    ... up to month_{duration}
  }},
  "monthly_percentages": {{
    "month_1": <pct>,
    "month_2": <pct>,
    ...
  }},
  "peak_month": <month_number>,
  "lead_time_months": <lead_time>,
  "order_month": <negative_month>,
  "vendor_strategy": "<strategy>",
  "risk_mitigation": "<mitigation_strategy>"
}}

Ensure monthly percentages sum to 100% and quantities match the total.
Output ONLY the JSON object, no other text."""

    return prompt


def create_all_procurement_prompts(bom_df, project_inputs: Dict) -> List[Dict]:
    """Create individual procurement prompts for each material from BoM"""
    prompts = []
    duration = project_inputs.get('procurement_duration', 18)

    material_category_map = {}
    for mat_name, mat_data in BASELINE_MATERIALS_220KV.items():
        material_category_map[mat_name.lower()] = mat_data['category']

    for _, row in bom_df.iterrows():
        row_dict = row.to_dict()
        material_name = row_dict.get('Material_Name',
                                     row_dict.get('material_name', ''))
        quantity_str = str(
            row_dict.get('Estimated_Quantity',
                         row_dict.get('estimated_quantity', 0)))
        unit = row_dict.get('Unit', row_dict.get('unit', ''))

        try:
            if '-' in quantity_str:
                parts = quantity_str.split('-')
                quantity = (float(parts[0]) + float(parts[1])) / 2
            else:
                clean_qty = re.sub(r'[^\d.]', '', quantity_str)
                quantity = float(clean_qty) if clean_qty else 0
        except:
            quantity = 0

        category = 'hardware'
        mat_lower = material_name.lower()
        for key, cat in material_category_map.items():
            if any(word in mat_lower for word in key.split()):
                category = cat
                break

        if 'steel' in mat_lower and 'rebar' not in mat_lower:
            category = 'steel'
        elif 'conductor' in mat_lower or 'acsr' in mat_lower:
            category = 'conductor'
        elif 'insulator' in mat_lower:
            category = 'insulator'
        elif 'concrete' in mat_lower or 'foundation' in mat_lower:
            category = 'concrete'

        prompt = create_single_material_procurement_prompt(
            material_name, quantity, unit, category, project_inputs)

        prompts.append({
            'material_name': material_name,
            'quantity': quantity,
            'unit': unit,
            'category': category,
            'prompt': prompt,
            'duration': duration
        })

    return prompts


def process_parallel_procurement_generation(
        prompts: List[Dict],
        model,
        max_workers: int = 20,
        progress_callback: Callable = None) -> Dict:
    """Process procurement prompts in parallel and aggregate results"""
    monthly_schedules = []
    total = len(prompts)

    def generate_for_material(prompt_data: Dict) -> Dict:
        try:
            response = model.generate_content(prompt_data['prompt'])
            response_text = response.text.strip()

            if response_text.startswith('```'):
                response_text = re.sub(r'^```json?\s*|\s*```$',
                                       '',
                                       response_text,
                                       flags=re.MULTILINE)

            result = json.loads(response_text)
            return {
                'success': True,
                'material_name': prompt_data['material_name'],
                'data': result
            }
        except Exception as e:
            duration = prompt_data.get('duration', 18)
            return {
                'success': False,
                'material_name': prompt_data['material_name'],
                'error': str(e),
                'data': {
                    'material': prompt_data['material_name'],
                    'category': prompt_data['category'],
                    'unit': prompt_data['unit'],
                    'total_quantity': prompt_data['quantity'],
                    'monthly_distribution': {
                        f"month_{i}": prompt_data['quantity'] / duration
                        for i in range(1, duration + 1)
                    },
                    'monthly_percentages': {
                        f"month_{i}": 100 / duration
                        for i in range(1, duration + 1)
                    },
                    'peak_month': 3,
                    'lead_time_months': 3,
                    'order_month': -3,
                    'vendor_strategy': 'Default strategy',
                    'risk_mitigation': 'Standard risk mitigation'
                }
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(generate_for_material, p): p
            for p in prompts
        }

        for idx, future in enumerate(as_completed(future_to_prompt), 1):
            result = future.result()
            monthly_schedules.append(result['data'])
            if progress_callback:
                progress_callback(idx, total, result['material_name'],
                                  result['success'])

    aggregated_result = {
        'monthly_schedule': monthly_schedules,
        'procurement_calendar':
        generate_procurement_calendar(monthly_schedules),
        'cost_optimization': generate_cost_optimization_tips(prompts),
        'risk_mitigation': generate_risk_mitigation(prompts),
        'summary': {
            'total_materials':
            len(monthly_schedules),
            'procurement_duration_months':
            18,
            'critical_path_items': [
                s['material'] for s in monthly_schedules
                if s.get('category') in ['steel', 'conductor']
            ],
            'long_lead_items': [
                s['material'] for s in monthly_schedules
                if s.get('lead_time_months', 0) >= 6
            ],
            'just_in_time_items': [
                s['material'] for s in monthly_schedules
                if s.get('lead_time_months', 0) <= 3
            ]
        }
    }

    return aggregated_result


def generate_procurement_calendar(schedules: List[Dict]) -> Dict:
    """Generate procurement calendar from individual schedules"""
    immediate = []
    short_term = []
    just_in_time = []

    for s in schedules:
        lead_time = s.get('lead_time_months', 3)
        item = {
            'material': s.get('material', ''),
            'quantity': f"{s.get('total_quantity', 0)} {s.get('unit', '')}",
            'lead_time': f"{lead_time} months",
            'order_by': f"Month -{lead_time}",
            'vendor_strategy': s.get('vendor_strategy', 'Standard procurement')
        }

        if lead_time >= 6:
            immediate.append(item)
        elif lead_time >= 3:
            short_term.append(item)
        else:
            just_in_time.append(item)

    return {
        'immediate_orders': immediate,
        'short_term_orders': short_term,
        'just_in_time_orders': just_in_time
    }


def generate_cost_optimization_tips(prompts: List[Dict]) -> Dict:
    """Generate cost optimization recommendations"""
    return {
        'bulk_discounts': [{
            'material': 'Steel',
            'threshold_quantity': '500 MT',
            'expected_discount': '5-8%'
        }, {
            'material': 'Conductors',
            'threshold_quantity': '200 MT',
            'expected_discount': '6-10%'
        }],
        'regional_vendors': [{
            'material_category':
            'Concrete',
            'recommendation':
            'Local RMC suppliers within 50km radius'
        }, {
            'material_category':
            'Hardware',
            'recommendation':
            'Regional fastener manufacturers'
        }]
    }


def generate_risk_mitigation(prompts: List[Dict]) -> Dict:
    """Generate risk mitigation strategies"""
    return {
        'price_volatility': [{
            'material': 'Steel',
            'risk_level': 'High',
            'mitigation': 'Forward contracts for 6-12 months'
        }, {
            'material': 'Conductors',
            'risk_level': 'Medium',
            'mitigation': 'LME-linked contracts with caps'
        }],
        'buffer_stock': [{
            'material': 'Steel',
            'buffer_percentage': '15%'
        }, {
            'material': 'Conductors',
            'buffer_percentage': '10%'
        }, {
            'material': 'Hardware',
            'buffer_percentage': '20%'
        }]
    }


def create_enhanced_bom_prompt(project_inputs):
    """
    Generate a comprehensive prompt for BoM generation using RAG approach
    (Kept for backward compatibility - single prompt version)
    """

    voltage = project_inputs['project_type'].split()[0]
    terrain = project_inputs['terrain']
    line_length = project_inputs.get('line_length', 100)
    scale_factor = line_length / 100

    voltage_key = f"{voltage}kV"
    voltage_mult = VOLTAGE_MULTIPLIERS.get(voltage_key,
                                           VOLTAGE_MULTIPLIERS["220kV"])
    terrain_mult = TERRAIN_MULTIPLIERS.get(terrain,
                                           TERRAIN_MULTIPLIERS["Plain"])

    baseline_table = "| Material Name | Unit | Quantity Range | IS Standards | Lead Time | Category | Priority |\n"
    baseline_table += "|---|---|---|---|---|---|---|\n"

    for material, data in BASELINE_MATERIALS_220KV.items():
        baseline_table += f"| {material} | {data['unit']} | {data['qty_range']} | {data['specs']} | {data['lead_time']} | {data['category']} | {data['priority']} |\n"

    prompt = f"""
You are a POWERGRID Senior Engineer with 20+ years of experience in transmission line projects.

**YOUR TASK:** Generate an accurate Bill of Materials for this project using engineering calculations.

**PROJECT SPECIFICATIONS:**
- **Project Name:** {project_inputs['project_name']}
- **Type:** {project_inputs['project_type']}
- **Voltage Level:** {voltage}kV
- **Line Length:** {line_length} km (Scale Factor: {scale_factor}x from 100km baseline)
- **Terrain:** {terrain}
- **Location:** {project_inputs['state']}, {project_inputs['region']}
- **Procurement Duration:** {project_inputs['procurement_duration']} months

**BASELINE REFERENCE DATA (220kV, 100km, Plain Terrain):**

{baseline_table}

**VOLTAGE ADJUSTMENT RULES:**

**{voltage_key} Specifications:**
- Description: {voltage_mult['description']}
- Steel Multiplier: {voltage_mult['steel_mult']}x
- Concrete Multiplier: {voltage_mult['concrete_mult']}x
- Insulators Multiplier: {voltage_mult['insulators_mult']}x
- Tower Count Multiplier: {voltage_mult['tower_count_mult']}x
- Notes: {voltage_mult['notes']}

**TERRAIN ADJUSTMENT RULES:**

**{terrain} Terrain Specifications:**
- Description: {terrain_mult['description']}
- Steel Multiplier: {terrain_mult['steel_mult']}x
- Concrete Multiplier: {terrain_mult['concrete_mult']}x
- Rebar Multiplier: {terrain_mult['rebar_mult']}x
- Engineering Reason: {terrain_mult['engineering_reason']}

**CALCULATION METHODOLOGY:**

```
Final Quantity = Baseline Quantity x Scale Factor x Voltage Multiplier x Terrain Multiplier

For each material:
1. Take baseline quantity (average if range given)
2. Multiply by scale factor ({scale_factor})
3. Apply voltage multiplier ({voltage_mult['steel_mult']} for steel, {voltage_mult['concrete_mult']} for concrete, etc.)
4. Apply terrain multiplier ({terrain_mult['steel_mult']} for steel, {terrain_mult['concrete_mult']} for concrete, etc.)
5. Round to appropriate precision
```

{EXAMPLE_CALCULATIONS}

**VALIDATION RULES:**
- Steel quantities should be in 500-5000 MT range for typical projects
- Conductor length should roughly equal line_length x 3.3 (3 phases + sag factor)
- Insulator count should be ~80-100 per tower
- Concrete volume should be ~14-16 m3 per tower (adjusted for terrain)
- All quantities must be engineering-sound and realistic

**OUTPUT FORMAT (Exact Markdown Table):**

| Material_Name | Unit | Estimated_Quantity | Priority | Calculation_Method | Notes |
|---|---|---|---|---|---|
| Galvanized Lattice Steel (Towers) | MT | [calculated] | High | Baseline 615 x {scale_factor} x {voltage_mult['steel_mult']} x {terrain_mult['steel_mult']} = [result] | {terrain_mult['engineering_reason']} |
| High-Tensile Bolts, Nuts, Washers | Nos | [calculated] | High | Baseline 117,500 x {scale_factor} x 1.0 x 1.0 = [result] | For all structural joints |
| ... | ... | ... | ... | ... | ... |

**CRITICAL INSTRUCTIONS:**
1. Include ALL 14 materials from the baseline reference
2. Show calculation formula in the Calculation_Method column
3. Apply multipliers correctly based on material category (steel/concrete/conductor/hardware)
4. Conductors and hardware are NOT affected by terrain (use 1.0x terrain multiplier)
5. Round quantities appropriately: MT to nearest 5, Nos to nearest 100, km to nearest 5
6. Priority assignment: High (critical path), Medium (standard), Low (accessories)
7. Output ONLY the markdown table, no additional text before or after

EVEN AFTER CALCULATING THE TABLE, TRY TO THINK ABOUT WHETHER THE QUANTITY IS ACCURATE OR NOT FROM YOUR DOMAIN KNOWLEDGE
IF YOU THINK, THAT THE CALCULATED VALUE IS NOT AS THAT OF THE GENERAL QUANTITY, YOU CAN REPLACE IT WITH THE QUANTITY YOU THINK IS RIGHT
PROVIDE REASONING FOR THE SAME IN THE NOTES COLUMN

**Generate the complete BoM table now following the exact format above:**
"""

    return prompt


def create_procurement_analysis_prompt(bom_df, project_inputs):
    """
    Generate comprehensive procurement analysis with JSON output
    (Kept for backward compatibility - single prompt version)
    """

    bom_markdown = bom_df.to_markdown(index=False)
    duration = project_inputs['procurement_duration']

    s_curve_summary = ""
    for category, pattern in S_CURVE_PATTERNS.items():
        s_curve_summary += f"\n**{category.upper()} MATERIALS:**\n"
        s_curve_summary += f"Description: {pattern['description']}\n"
        for phase, details in pattern['phases'].items():
            s_curve_summary += f"- {phase.replace('_', ' ').title()}: {details['months']} months -> {details['percentage']} ({details['breakdown']})\n"
        s_curve_summary += f"- Lead Time: {pattern['lead_time']}\n"
        s_curve_summary += f"- Order Timing: {pattern['order_timing']}\n"
        s_curve_summary += f"- Strategy: {pattern['procurement_strategy']}\n"
        s_curve_summary += f"- Risk: {pattern['risk_level']}\n\n"

    prompt = f"""
You are a POWERGRID Procurement Strategist with expertise in transmission line material planning.

**YOUR TASK:** Create a detailed procurement schedule in JSON format using S-curve patterns.

**VERIFIED BILL OF MATERIALS:**
{bom_markdown}

**PROJECT CONTEXT:**
- Project Name: {project_inputs['project_name']}
- Procurement Duration: {duration} months
- Total Budget: Rs.{project_inputs['total_budget']} Crores
- Location: {project_inputs['state']}, {project_inputs['region']}
- Terrain: {project_inputs['terrain']}
- Monsoon Factor: {"Enabled" if project_inputs.get('monsoon_factor', False) else "Disabled"}

**S-CURVE PROCUREMENT PATTERNS (POWERGRID Industry Standard):**
{s_curve_summary}

{PROCUREMENT_BEST_PRACTICES}

**OUTPUT FORMAT - STRICT JSON:**

Output a valid JSON object with monthly_schedule array containing each material.

**CRITICAL INSTRUCTIONS:**
1. OUTPUT ONLY VALID JSON - No markdown code blocks, no explanatory text
2. Include ALL materials from the BoM in monthly_schedule array
3. Monthly distributions must sum to 100% for each material
4. Use actual material names from the BoM table
5. Follow S-curve patterns strictly

**Generate the complete procurement analysis JSON now:**
"""

    return prompt
