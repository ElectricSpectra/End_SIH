import requests
from bs4 import BeautifulSoup
import re
import time
from typing import Dict, List, Optional
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

class WebScrapingResearcher:
    """
    Enhanced web scraping researcher with parallel processing and raw HTML capture
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        self.material_categories = {
            'steel': ['steel', 'galvanized', 'angle', 'rebar', 'reinforcement'],
            'conductor': ['conductor', 'acsr', 'wire', 'cable', 'copper', 'aluminum'],
            'concrete': ['concrete', 'cement', 'm20', 'm25'],
            'insulator': ['insulator', 'disc', 'porcelain', 'composite'],
            'hardware': ['bolt', 'nut', 'washer', 'clamp', 'fastener'],
            'transformer': ['transformer', 'switchgear', 'breaker']
        }
        
        self.gst_rates = {
            'steel': 18, 'conductor': 18, 'concrete': 12,
            'insulator': 18, 'hardware': 18, 'transformer': 18, 'general': 18
        }
    
    def categorize_material(self, material_name: str) -> str:
        """Categorize material based on name"""
        material_lower = material_name.lower()
        for category, keywords in self.material_categories.items():
            for keyword in keywords:
                if keyword in material_lower:
                    return category
        return 'general'
    
    def scrape_material_context(self, material_name: str) -> Dict:
        """
        Scrape raw HTML context for a material (for Gemini analysis)
        Returns: {
            'material': str,
            'sources': [{'url': str, 'html_snippet': str, 'text_snippet': str}],
            'category': str
        }
        """
        category = self.categorize_material(material_name)
        sources = []
        
        # Build search queries
        search_queries = [
            f"https://www.indiamart.com/impcat/{material_name.replace(' ', '+')}.html",
        ]
        
        # Add category-specific sources
        if category == 'steel':
            search_queries.append("https://www.steelmint.com/prices")
        elif category == 'concrete':
            search_queries.extend([
                "https://www.ultratechcement.com",
                "https://www.ambujacement.com"
            ])
        
        for url in search_queries:
            try:
                response = requests.get(url, headers=self.headers, timeout=8)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract price-related sections
                    price_elements = soup.find_all(['div', 'span', 'p'], class_=re.compile(r'price|cost|rate', re.I))
                    
                    # Get text snippets (first 5 relevant sections)
                    text_snippets = []
                    for elem in price_elements[:5]:
                        text = elem.get_text(strip=True)
                        if re.search(r'₹|Rs\.?\s*\d', text):  # Contains price
                            text_snippets.append(text[:300])  # Limit length
                    
                    if text_snippets:
                        sources.append({
                            'url': url,
                            'text_snippet': ' | '.join(text_snippets),
                            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                # Small delay to avoid rate limiting
                time.sleep(random.uniform(0.3, 0.8))
                
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
                continue
        
        return {
            'material': material_name,
            'category': category,
            'sources': sources,
            'total_sources': len(sources)
        }
    
    def scrape_materials_parallel(self, materials: List[str], max_workers: int = 5) -> List[Dict]:
        """
        Scrape multiple materials in parallel
        Returns: List of scraping results
        """
        results = []
        
        print(f"🚀 Starting parallel scraping for {len(materials)} materials (max {max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_material = {
                executor.submit(self.scrape_material_context, material): material 
                for material in materials
            }
            
            # Collect results as they complete
            for idx, future in enumerate(as_completed(future_to_material), 1):
                material = future_to_material[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ [{idx}/{len(materials)}] Scraped {material} ({result['total_sources']} sources)")
                except Exception as e:
                    print(f"❌ [{idx}/{len(materials)}] Failed {material}: {e}")
                    results.append({
                        'material': material,
                        'category': self.categorize_material(material),
                        'sources': [],
                        'error': str(e)
                    })
        
        print(f"✅ Parallel scraping complete! Total sources: {sum(r['total_sources'] for r in results)}")
        return results
    
    def get_gst_rate(self, material_name: str) -> int:
        """Get applicable GST rate"""
        category = self.categorize_material(material_name)
        return self.gst_rates.get(category, 18)


def analyze_material_costs_with_gemini(materials: List[str], quantities: List[str], units: List[str], 
                                       region: str, state: str, gemini_model) -> Dict:
    """
    Main function using Gemini for price extraction from web-scraped data
    """
    researcher = WebScrapingResearcher()
    
    # Step 1: Parallel web scraping
    print("\n🔍 Step 1/2: Web scraping material prices...")
    scraped_data = researcher.scrape_materials_parallel(materials, max_workers=5)
    
    # Step 2: Send to Gemini for analysis
    print("\n🤖 Step 2/2: Gemini analyzing scraped data...")
    
    # Build comprehensive prompt for Gemini
    gemini_prompt = f"""You are a POWERGRID procurement cost analyst. Analyze web-scraped material price data and extract accurate unit prices.

**CRITICAL TASK:**
Extract price per unit for each material, ensuring unit conversion accuracy.

**PROJECT CONTEXT:**
- Region: {region}
- State: {state}
- Analysis Date: {time.strftime('%Y-%m-%d')}

**MATERIALS TO ANALYZE:**

"""
    
    # Add each material with its scraped data and required unit
    for idx, (material, qty_str, unit, scraped_item) in enumerate(zip(materials, quantities, units, scraped_data), 1):
        gemini_prompt += f"""
---
**Material {idx}: {material}**
- Required Quantity: {qty_str} {unit}
- Required Unit for Pricing: {unit}
- Category: {scraped_item.get('category', 'unknown')}

**Web-Scraped Price Data:**
"""
        
        if scraped_item.get('sources'):
            for source in scraped_item['sources']:
                gemini_prompt += f"\n- Source: {source['url']}\n"
                gemini_prompt += f"  Text: {source['text_snippet']}\n"
        else:
            gemini_prompt += "\n- No web data found (use market estimates)\n"
    
    gemini_prompt += f"""

---

**OUTPUT FORMAT (STRICT JSON):**

You MUST return a JSON array with this exact structure for each material:

```json
[
  {{
    "material": "Galvanized Lattice Steel (Towers)",
    "required_unit": "MT",
    "extracted_price_per_unit": 55000,
    "price_unit": "MT",
    "confidence": "High",
    "source_basis": "SteelMint avg price ₹55,000/MT",
    "price_trend": "Increasing (+2.3%)",
    "notes": "Multiple sources confirm range ₹52k-58k/MT"
  }},
  {{
    "material": "ACSR Zebra Conductor",
    "required_unit": "MT",
    "extracted_price_per_unit": 280000,
    "price_unit": "MT",
    "confidence": "Medium",
    "source_basis": "IndiaMART ₹280/kg → ₹280,000/MT",
    "price_trend": "Stable",
    "notes": "Converted from per-kg pricing"
  }},
  {{
    "material": "Concrete for Foundations (M20/25)",
    "required_unit": "Cum",
    "extracted_price_per_unit": 3500,
    "price_unit": "Cum",
    "confidence": "Medium",
    "source_basis": "RMC suppliers ₹3,500/cum",
    "price_trend": "Stable (+1%)",
    "notes": "Regional RMC plant pricing"
  }}
]
```

**RULES:**
1. **Unit Conversion:** If scraped price is per kg, convert to MT (multiply by 1000)
2. **Price Validation:** Check if price is realistic for {state}, {region}
3. **Confidence Levels:**
   - High: Multiple sources, clear pricing
   - Medium: Single source or conversion needed
   - Low: Estimated based on category
4. **Trend Detection:** Look for keywords like "increased", "stable", "decreased"
5. **All prices in INR (Indian Rupees)**
6. **DO NOT use markdown code blocks - output raw JSON only**

**RESPOND WITH ONLY THE JSON ARRAY, NO OTHER TEXT:**
"""
    
    try:
        # Call Gemini
        response = gemini_model.generate_content(gemini_prompt)
        
        # Parse JSON response
        import json
        # Remove markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith('```'):
            response_text = re.sub(r'^```json?\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        
        gemini_prices = json.loads(response_text)
        
        # Process results
        material_costs = []
        total_cost = 0
        
        for price_data, qty_str, unit in zip(gemini_prices, quantities, units):
            # Parse quantity
            try:
                if '-' in str(qty_str):
                    qty_parts = str(qty_str).split('-')
                    quantity = (float(qty_parts[0]) + float(qty_parts[1])) / 2
                else:
                    clean_qty = re.sub(r'[^\d.]', '', str(qty_str))
                    quantity = float(clean_qty) if clean_qty else 0
            except:
                quantity = 0
            
            unit_price = price_data.get('extracted_price_per_unit', 0)
            
            # Calculate cost
            total_material_cost = quantity * unit_price
            total_material_cost_crores = total_material_cost / 10000000
            
            # GST
            gst_rate = researcher.get_gst_rate(price_data['material'])
            gst_amount = total_material_cost_crores * (gst_rate / 100)
            cost_with_gst = total_material_cost_crores + gst_amount
            
            material_costs.append({
                'material': price_data['material'],
                'quantity': quantity,
                'unit': unit,
                'unit_price': unit_price,
                'base_cost_cr': round(total_material_cost_crores, 4),
                'gst_rate': f"{gst_rate}%",
                'gst_amount_cr': round(gst_amount, 4),
                'total_cost_cr': round(cost_with_gst, 4),
                'price_trend': price_data.get('price_trend', 'Unknown'),
                'source': price_data.get('source_basis', 'Gemini Analysis'),
                'confidence': price_data.get('confidence', 'Medium'),
                'notes': price_data.get('notes', '')
            })
            
            total_cost += cost_with_gst
        
        # Calculate monthly distribution (simplified S-curve)
        s_curve_distribution = [
            0.05, 0.08, 0.12, 0.15, 0.18, 0.12,
            0.08, 0.06, 0.05, 0.04, 0.03, 0.02,
            0.01, 0.005, 0.003, 0.002, 0.001, 0.001
        ]
        
        monthly_costs = {}
        for month_idx, distribution in enumerate(s_curve_distribution):
            month_name = f"Month {month_idx + 1}"
            monthly_costs[month_name] = round(total_cost * distribution, 4)
        
        # GST summary
        gst_summary = []
        for item in material_costs:
            gst_summary.append({
                'material': item['material'],
                'gst_rate': item['gst_rate'],
                'taxable_amount_cr': item['base_cost_cr'],
                'gst_amount_cr': item['gst_amount_cr']
            })
        
        return {
            'total_cost': round(total_cost, 2),
            'material_costs': material_costs,
            'monthly_costs': monthly_costs,
            'gst_summary': gst_summary,
            'region': region,
            'state': state,
            'analysis_date': time.strftime('%Y-%m-%d'),
            'gemini_analysis': True
        }
        
    except Exception as e:
        print(f"❌ Gemini analysis failed: {e}")
        raise


# Test function
if __name__ == "__main__":
    researcher = WebScrapingResearcher()
    
    test_materials = ["Galvanized Steel", "ACSR Conductor"]
    results = researcher.scrape_materials_parallel(test_materials, max_workers=3)
    
    for result in results:
        print(f"\n{result['material']}: {result['total_sources']} sources")
        for source in result['sources']:
            print(f"  - {source['url'][:50]}...")