"""
Vendor Analysis System
Implements three core principles:
1. Calculate True Cost of Ownership (TCO)
2. Compute Vendor Reliability Scores
3. Build Vendor Ecosystem Map (Network Graph)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import json
import os

# Optional geopy import for distance calculation
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False


class VendorAnalyzer:
    """Comprehensive vendor analysis system based on three principles."""
    
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with vendor data.
        
        Args:
            csv_file_path: Path to the vendors CSV file
        """
        self.df = pd.read_csv(csv_file_path)
        self.tco_scores = None
        self.reliability_scores = None
        self.ecosystem_map = None
        self.final_rankings = None
        
    # ============== PRINCIPLE 1: TRUE COST OF OWNERSHIP (TCO) ==============
    
    def calculate_tco(self):
        """
        Calculate True Cost of Ownership for each vendor.
        
        TCO Formula:
        TCO = Base_Cost + Holding_Cost + Lead_Time_Cost + Logistics_Cost
        
        Where:
        - Base_Cost: Price per MT with GST adjustment
        - Holding_Cost: Based on capacity utilization
        - Lead_Time_Cost: Penalty for longer lead times
        - Logistics_Cost: Estimated based on distance factor
        
        Returns:
            pd.DataFrame with TCO calculations
        """
        df = self.df.copy()
        
        # 1. Base Cost with GST
        df['base_cost'] = df['current_price_per_mt'] * (1 + df['gst_pct'] / 100)
        
        # 2. Holding Cost (inventory carrying cost as % of price)
        # Larger capacity means better inventory flexibility
        df['holding_cost_pct'] = 5.0  # Base holding cost 5%
        df.loc[df['capacity_mt_yr'] < 5000, 'holding_cost_pct'] = 8.0
        df.loc[df['capacity_mt_yr'] > 20000, 'holding_cost_pct'] = 3.0
        df['holding_cost'] = df['base_cost'] * (df['holding_cost_pct'] / 100)
        
        # 3. Lead Time Cost (longer lead time = higher cost)
        # Normalized to cost impact
        max_lead_time = df['lead_time_days'].max()
        df['lead_time_cost'] = (df['lead_time_days'] / max_lead_time) * df['base_cost'] * 0.1
        
        # 4. Logistics Cost (based on state, normalized)
        # Distance factor varies by state (simplified)
        state_logistics_factor = {
            'Jharkhand': 1.0, 'Karnataka': 1.1, 'Maharashtra': 0.95, 'Gujarat': 1.05,
            'Rajasthan': 1.15, 'Tamil Nadu': 1.2, 'Uttar Pradesh': 1.0, 'Haryana': 0.9,
            'West Bengal': 1.05, 'Punjab': 1.1, 'Delhi': 0.85, 'Odisha': 1.15
        }
        df['logistics_factor'] = df['state'].map(state_logistics_factor).fillna(1.0)
        df['logistics_cost'] = df['base_cost'] * 0.05 * df['logistics_factor']
        
        # 5. Discount Benefit (bulk discounts reduce TCO)
        df['discount_benefit'] = df['base_cost'] * (df['bulk_discount_pct'] / 100)
        
        # Total TCO per MT
        df['tco_per_mt'] = (
            df['base_cost'] + 
            df['holding_cost'] + 
            df['lead_time_cost'] + 
            df['logistics_cost'] - 
            df['discount_benefit']
        )
        
        # TCO Score (lower is better, so invert to 0-100 scale)
        min_tco = df['tco_per_mt'].min()
        max_tco = df['tco_per_mt'].max()
        df['tco_score'] = 100 - ((df['tco_per_mt'] - min_tco) / (max_tco - min_tco)) * 100
        
        self.tco_scores = df[['vendor_id', 'vendor_name', 'tco_per_mt', 'tco_score']]
        return self.tco_scores
    
    # ============== PRINCIPLE 2: VENDOR RELIABILITY SCORES ==============
    
    def compute_reliability_scores(self):
        """
        Compute Vendor Reliability Scores based on multiple factors.
        
        Factors:
        1. Capacity Reliability: Larger capacity = more reliable
        2. Lead Time Consistency: Shorter lead times = more reliable
        3. Discount Stability: Competitive discounts = better deals
        4. GST Compliance: Lower GST = better financial terms
        5. Geographic Spread: Better distribution = accessibility
        
        Returns:
            pd.DataFrame with reliability scores
        """
        df = self.df.copy()
        
        # 1. Capacity Reliability Score (0-30 points)
        # Larger capacity indicates reliability
        capacity_normalized = (df['capacity_mt_yr'] - df['capacity_mt_yr'].min()) / \
                             (df['capacity_mt_yr'].max() - df['capacity_mt_yr'].min())
        df['capacity_score'] = capacity_normalized * 30
        
        # 2. Lead Time Consistency Score (0-25 points)
        # Shorter lead time is better
        lead_time_normalized = 1 - ((df['lead_time_days'] - df['lead_time_days'].min()) / \
                                    (df['lead_time_days'].max() - df['lead_time_days'].min()))
        df['lead_time_score'] = lead_time_normalized * 25
        
        # 3. Discount Stability Score (0-20 points)
        # Higher discounts indicate competitive pricing
        discount_normalized = df['bulk_discount_pct'] / df['bulk_discount_pct'].max()
        df['discount_score'] = discount_normalized * 20
        
        # 4. GST Compliance Score (0-15 points)
        # Lower GST is favorable
        gst_normalized = 1 - ((df['gst_pct'] - df['gst_pct'].min()) / \
                              (df['gst_pct'].max() - df['gst_pct'].min()))
        df['gst_score'] = gst_normalized * 15
        
        # 5. Vendor Category Specialization Score (0-10 points)
        # Vendor divisions might indicate specialization
        df['is_specialized'] = df['vendor_name'].str.contains('Div', case=False, na=False).astype(int)
        df['specialization_score'] = (1 - df['is_specialized']) * 10  # Main vendors get higher score
        
        # Total Reliability Score (0-100)
        df['reliability_score'] = (
            df['capacity_score'] + 
            df['lead_time_score'] + 
            df['discount_score'] + 
            df['gst_score'] + 
            df['specialization_score']
        )
        
        self.reliability_scores = df[[
            'vendor_id', 'vendor_name', 'material_category', 'state',
            'capacity_score', 'lead_time_score', 'discount_score',
            'gst_score', 'specialization_score', 'reliability_score'
        ]]
        return self.reliability_scores
    
    # ============== PRINCIPLE 3: VENDOR ECOSYSTEM MAP ==============
    
    def build_ecosystem_map(self):
        """
        Build Vendor Ecosystem Map (Network Graph).
        
        Creates a network showing:
        1. Vendor relationships based on material category
        2. Geographic clustering (state-based)
        3. Price competitiveness clusters
        4. Capacity-based partnerships
        
        Returns:
            networkx.Graph: Vendor ecosystem network
        """
        G = nx.Graph()
        
        # Add nodes for each vendor with attributes
        for _, row in self.df.iterrows():
            G.add_node(
                row['vendor_id'],
                vendor_name=row['vendor_name'],
                category=row['material_category'],
                state=row['state'],
                capacity=row['capacity_mt_yr'],
                price=row['current_price_per_mt'],
                lead_time=row['lead_time_days']
            )
        
        # Create edges based on relationships
        # 1. Same material category vendors (can form supply chain partnerships)
        category_vendors = defaultdict(list)
        for _, row in self.df.iterrows():
            category_vendors[row['material_category']].append(row['vendor_id'])
        
        for category, vendors in category_vendors.items():
            for i, vendor1 in enumerate(vendors):
                for vendor2 in vendors[i+1:]:
                    G.add_edge(vendor1, vendor2, relationship='same_category', weight=1)
        
        # 2. Geographic proximity (same state vendors)
        state_vendors = defaultdict(list)
        for _, row in self.df.iterrows():
            state_vendors[row['state']].append(row['vendor_id'])
        
        for state, vendors in state_vendors.items():
            for i, vendor1 in enumerate(vendors):
                for vendor2 in vendors[i+1:]:
                    if G.has_edge(vendor1, vendor2):
                        G[vendor1][vendor2]['weight'] += 1
                    else:
                        G.add_edge(vendor1, vendor2, relationship='geographic', weight=0.5)
        
        # 3. Price competitiveness clusters (similar pricing)
        df_sorted = self.df.sort_values('current_price_per_mt')
        for i in range(len(df_sorted) - 1):
            vendor1 = df_sorted.iloc[i]['vendor_id']
            vendor2 = df_sorted.iloc[i+1]['vendor_id']
            price_diff_pct = abs(df_sorted.iloc[i]['current_price_per_mt'] - 
                                df_sorted.iloc[i+1]['current_price_per_mt']) / \
                            df_sorted.iloc[i]['current_price_per_mt'] * 100
            
            if price_diff_pct < 5:  # Within 5% price range
                if G.has_edge(vendor1, vendor2):
                    G[vendor1][vendor2]['weight'] += 0.5
                else:
                    G.add_edge(vendor1, vendor2, relationship='price_competitive', weight=0.5)
        
        self.ecosystem_map = G
        return G
    
    def visualize_ecosystem(self, output_file='ecosystem_map.png'):
        """
        Visualize the vendor ecosystem network.
        
        Args:
            output_file: Output filename for the visualization
        """
        if self.ecosystem_map is None:
            print("Ecosystem map not built yet. Call build_ecosystem_map() first.")
            return
        
        G = self.ecosystem_map
        
        # Create figure with better size
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Color nodes by material category
        category_colors = {
            'Concrete': '#FF6B6B',
            'Conductor': '#4ECDC4',
            'Hardware': '#45B7D1',
            'Insulator': '#FFA07A',
            'Steel': '#98D8C8'
        }
        
        node_colors = [
            category_colors.get(G.nodes[node].get('category', 'Unknown'), '#CCCCCC')
            for node in G.nodes()
        ]
        
        # Node sizes based on capacity
        node_sizes = [
            (G.nodes[node].get('capacity', 1000) / 100) + 100
            for node in G.nodes()
        ]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, width=[w/max_weight * 3 for w in weights], 
                              alpha=0.3)
        
        # Draw labels (using vendor_id)
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=category)
            for category, color in category_colors.items()
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.title('Vendor Ecosystem Map\n(Node size = Capacity, Edge thickness = Relationship strength)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Ecosystem map saved to {output_file}")
        plt.close()
    
    def get_ecosystem_statistics(self):
        """
        Get statistics about the vendor ecosystem.
        
        Returns:
            dict: Ecosystem statistics
        """
        if self.ecosystem_map is None:
            return {}
        
        G = self.ecosystem_map
        stats = {
            'total_vendors': G.number_of_nodes(),
            'total_relationships': G.number_of_edges(),
            'avg_connections_per_vendor': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'network_density': nx.density(G),
            'largest_component_size': len(max(nx.connected_components(G), key=len)),
            'avg_clustering_coefficient': nx.average_clustering(G) if G.number_of_nodes() > 1 else 0
        }
        return stats
    
    # ============== FINAL SELECTION: TOP 3 VENDORS ==============
    
    def select_top_vendors(self, material_category=None, weight_tco=0.4, weight_reliability=0.4, weight_ecosystem=0.2):
        """
        Select top 3 vendors based on combined scoring of all three principles.
        
        Args:
            material_category: Optional filter for material category (e.g., 'Steel', 'Concrete')
            weight_tco: Weight for TCO score (0-1)
            weight_reliability: Weight for reliability score (0-1)
            weight_ecosystem: Weight for ecosystem/network score (0-1)
        
        Returns:
            pd.DataFrame: Top 3 vendors with detailed analysis
        """
        # Ensure all scores are calculated
        if self.tco_scores is None:
            self.calculate_tco()
        if self.reliability_scores is None:
            self.compute_reliability_scores()
        if self.ecosystem_map is None:
            self.build_ecosystem_map()
        
        # Merge scores
        df = self.df.copy()
        
        # Filter by material category if specified
        if material_category:
            df = df[df['material_category'].str.lower() == material_category.lower()]
            if len(df) == 0:
                return pd.DataFrame()  # Return empty if no vendors in category
        
        df = df.merge(self.tco_scores[['vendor_id', 'tco_score']], on='vendor_id')
        df = df.merge(self.reliability_scores[['vendor_id', 'reliability_score']], on='vendor_id')
        
        # Calculate ecosystem score based on network centrality
        G = self.ecosystem_map
        betweenness_centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)
        
        ecosystem_scores = {}
        for vendor_id in df['vendor_id']:
            bc = betweenness_centrality.get(vendor_id, 0)
            dc = degree_centrality.get(vendor_id, 0)
            # Combine centrality measures
            ecosystem_scores[vendor_id] = (bc * 0.6 + dc * 0.4) * 100
        
        df['ecosystem_score'] = df['vendor_id'].map(ecosystem_scores).fillna(0)
        
        # Normalize scores to 0-100 scale
        for score_col in ['tco_score', 'reliability_score', 'ecosystem_score']:
            min_val = df[score_col].min()
            max_val = df[score_col].max()
            if max_val > min_val:
                df[score_col] = (df[score_col] - min_val) / (max_val - min_val) * 100
        
        # Calculate combined score
        df['combined_score'] = (
            weight_tco * df['tco_score'] +
            weight_reliability * df['reliability_score'] +
            weight_ecosystem * df['ecosystem_score']
        )
        
        # Get top 3 vendors
        top_3 = df.nlargest(3, 'combined_score')[
            ['vendor_id', 'vendor_name', 'material_category', 'state',
             'current_price_per_mt', 'capacity_mt_yr', 'lead_time_days',
             'bulk_discount_pct', 'gst_pct', 'tco_score', 'reliability_score', 
             'ecosystem_score', 'combined_score']
        ].reset_index(drop=True)
        
        self.final_rankings = top_3
        return top_3
    
    # ============== COMPREHENSIVE ANALYSIS REPORT ==============
    
    def generate_analysis_report(self):
        """
        Generate a comprehensive analysis report with all three principles.
        
        Returns:
            dict: Complete analysis report
        """
        report = {
            'tco_analysis': self.calculate_tco().to_dict(orient='records'),
            'reliability_analysis': self.compute_reliability_scores().to_dict(orient='records'),
            'ecosystem_statistics': self.get_ecosystem_statistics(),
            'top_3_vendors': self.select_top_vendors().to_dict(orient='records')
        }
        return report
    
    def print_analysis_summary(self):
        """Print a formatted summary of the vendor analysis."""
        print("\n" + "="*100)
        print("VENDOR ANALYSIS REPORT - THREE PRINCIPLES")
        print("="*100)
        
        # 1. TCO Analysis Summary
        print("\n1. TRUE COST OF OWNERSHIP (TCO) ANALYSIS")
        print("-" * 100)
        tco_df = self.calculate_tco().sort_values('tco_score', ascending=False).head(5)
        print("Top 5 Vendors by TCO Score (lower TCO is better):")
        print(tco_df.to_string(index=False))
        
        # 2. Reliability Analysis Summary
        print("\n2. VENDOR RELIABILITY SCORES")
        print("-" * 100)
        rel_df = self.compute_reliability_scores().sort_values('reliability_score', ascending=False).head(5)
        print("Top 5 Vendors by Reliability Score:")
        print(rel_df[['vendor_id', 'vendor_name', 'reliability_score']].to_string(index=False))
        
        # 3. Ecosystem Analysis Summary
        print("\n3. VENDOR ECOSYSTEM MAP ANALYSIS")
        print("-" * 100)
        self.build_ecosystem_map()
        eco_stats = self.get_ecosystem_statistics()
        print(f"Total Vendors in Ecosystem: {eco_stats['total_vendors']}")
        print(f"Total Relationships: {eco_stats['total_relationships']}")
        print(f"Average Connections per Vendor: {eco_stats['avg_connections_per_vendor']:.2f}")
        print(f"Network Density: {eco_stats['network_density']:.4f}")
        print(f"Clustering Coefficient: {eco_stats['avg_clustering_coefficient']:.4f}")
        
        # 4. Final Selection
        print("\n4. TOP 3 SELECTED VENDORS (Combined Score)")
        print("-" * 100)
        self.build_ecosystem_map()
        top_vendors = self.select_top_vendors()
        print(top_vendors[['vendor_id', 'vendor_name', 'material_category', 'state', 
                          'combined_score', 'tco_score', 'reliability_score', 
                          'ecosystem_score']].to_string(index=False))
        
        print("\n" + "="*100)


# ============== PROJECT-BASED VENDOR SELECTION (BOM INTEGRATED) ==============

class BOMBasedVendorAnalyzer(VendorAnalyzer):
    """
    Extended VendorAnalyzer that works with project details and BOM data.
    Considers transport costs, material quantities, and project location.
    """
    
    def __init__(self, csv_file_path, project_inputs=None, bom_df=None):
        """
        Initialize with vendor data and optional project context.
        
        Args:
            csv_file_path: Path to vendors CSV
            project_inputs: Dict with project details (state, region, etc.)
            bom_df: DataFrame with Bill of Materials (Material_Name, Estimated_Quantity, Unit, etc.)
        """
        super().__init__(csv_file_path)
        self.project_inputs = project_inputs or {}
        self.bom_df = bom_df
        self.transport_costs = {}
        self.material_mapping = {}
        
    def estimate_transport_cost(self, vendor_state, project_state, quantity, distance_km=None):
        """
        Estimate transport cost based on distance and quantity.
        
        Args:
            vendor_state: Vendor's state
            project_state: Project location state
            quantity: Material quantity in MT
            distance_km: Optional distance; if not provided, estimated from state
        
        Returns:
            float: Transport cost as percentage of material cost
        """
        # State to approximate location (lat, long)
        state_locations = {
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
        }
        
        # Estimate distance if not provided
        if distance_km is None:
            if vendor_state in state_locations and project_state in state_locations:
                vendor_loc = state_locations[vendor_state]
                project_loc = state_locations[project_state]
                
                if GEOPY_AVAILABLE:
                    try:
                        distance_km = geodesic(vendor_loc, project_loc).km
                    except:
                        distance_km = abs(vendor_loc[0] - project_loc[0]) * 111  # Rough estimate
                else:
                    # Rough estimation: 111 km per degree latitude
                    distance_km = abs(vendor_loc[0] - project_loc[0]) * 111
            else:
                distance_km = 500  # Default average distance
        
        # Transport cost calculation
        # Base rate: ₹5-8 per MT per km depending on material
        base_rate_per_mt_km = 6  # Average
        
        # Adjust for quantity (bulk discounts on transport)
        if quantity > 500:
            base_rate_per_mt_km *= 0.85
        elif quantity > 1000:
            base_rate_per_mt_km *= 0.75
        
        total_transport_cost = distance_km * quantity * base_rate_per_mt_km
        return total_transport_cost
    
    def calculate_tco_with_bom(self):
        """
        Calculate TCO including transport costs based on actual BOM quantities.
        
        Returns:
            pd.DataFrame with TCO calculations including transport
        """
        df = self.df.copy()
        project_state = self.project_inputs.get('state', 'Maharashtra')
        
        # 1. Base Cost with GST
        df['base_cost'] = df['current_price_per_mt'] * (1 + df['gst_pct'] / 100)
        
        # 2. Holding Cost
        df['holding_cost_pct'] = 5.0
        df.loc[df['capacity_mt_yr'] < 5000, 'holding_cost_pct'] = 8.0
        df.loc[df['capacity_mt_yr'] > 20000, 'holding_cost_pct'] = 3.0
        df['holding_cost'] = df['base_cost'] * (df['holding_cost_pct'] / 100)
        
        # 3. Lead Time Cost
        max_lead_time = df['lead_time_days'].max()
        df['lead_time_cost'] = (df['lead_time_days'] / max_lead_time) * df['base_cost'] * 0.1
        
        # 4. Logistics Cost (state-based)
        state_logistics_factor = {
            'Jharkhand': 1.0, 'Karnataka': 1.1, 'Maharashtra': 0.95, 'Gujarat': 1.05,
            'Rajasthan': 1.15, 'Tamil Nadu': 1.2, 'Uttar Pradesh': 1.0, 'Haryana': 0.9,
            'West Bengal': 1.05, 'Punjab': 1.1, 'Delhi': 0.85, 'Odisha': 1.15
        }
        df['logistics_factor'] = df['state'].map(state_logistics_factor).fillna(1.0)
        df['logistics_cost'] = df['base_cost'] * 0.05 * df['logistics_factor']
        
        # 5. Transport Cost (based on BOM quantities if available)
        df['transport_cost'] = 0.0
        if self.bom_df is not None:
            for idx, row in df.iterrows():
                vendor_id = row['vendor_id']
                vendor_state = row['state']
                
                # Find relevant BOM items for this vendor's category
                matching_bom = self.bom_df[
                    self.bom_df['Material_Name'].str.lower().str.contains(
                        row['material_category'].lower(), na=False
                    )
                ]
                
                if len(matching_bom) > 0:
                    total_qty = matching_bom['Estimated_Quantity'].sum()
                    transport = self.estimate_transport_cost(vendor_state, project_state, total_qty)
                    # Normalize transport cost per MT for scoring
                    df.loc[idx, 'transport_cost'] = transport / max(total_qty, 1) * 0.01
        
        # 6. Discount Benefit
        df['discount_benefit'] = df['base_cost'] * (df['bulk_discount_pct'] / 100)
        
        # Total TCO per MT
        df['tco_per_mt'] = (
            df['base_cost'] + 
            df['holding_cost'] + 
            df['lead_time_cost'] + 
            df['logistics_cost'] + 
            df['transport_cost'] - 
            df['discount_benefit']
        )
        
        # TCO Score
        min_tco = df['tco_per_mt'].min()
        max_tco = df['tco_per_mt'].max()
        df['tco_score'] = 100 - ((df['tco_per_mt'] - min_tco) / (max_tco - min_tco)) * 100
        
        self.tco_scores = df[[
            'vendor_id', 'vendor_name', 'tco_per_mt', 'transport_cost', 'tco_score'
        ]]
        return self.tco_scores
    
    def compute_reliability_scores(self):
        """
        Compute reliability scores with adjustment for project location proximity.
        
        Returns:
            pd.DataFrame with reliability scores
        """
        df = self.df.copy()
        project_state = self.project_inputs.get('state', 'Maharashtra')
        
        # 1. Capacity Reliability Score (0-25 points)
        capacity_normalized = (df['capacity_mt_yr'] - df['capacity_mt_yr'].min()) / \
                             (df['capacity_mt_yr'].max() - df['capacity_mt_yr'].min())
        df['capacity_score'] = capacity_normalized * 25
        
        # 2. Lead Time Consistency Score (0-20 points)
        lead_time_normalized = 1 - ((df['lead_time_days'] - df['lead_time_days'].min()) / \
                                    (df['lead_time_days'].max() - df['lead_time_days'].min()))
        df['lead_time_score'] = lead_time_normalized * 20
        
        # 3. Discount Stability Score (0-15 points)
        discount_normalized = df['bulk_discount_pct'] / df['bulk_discount_pct'].max()
        df['discount_score'] = discount_normalized * 15
        
        # 4. Geographic Proximity Score (0-20 points) - NEW FOR BOM ANALYSIS
        df['geographic_score'] = 0.0
        for idx, row in df.iterrows():
            if row['state'].lower() == project_state.lower():
                df.loc[idx, 'geographic_score'] = 20.0
            elif row['state'] in ['Gujarat', 'Maharashtra', 'Karnataka']:
                df.loc[idx, 'geographic_score'] = 15.0
            else:
                df.loc[idx, 'geographic_score'] = 10.0
        
        # 5. GST Compliance Score (0-10 points)
        gst_normalized = 1 - ((df['gst_pct'] - df['gst_pct'].min()) / \
                              (df['gst_pct'].max() - df['gst_pct'].min()))
        df['gst_score'] = gst_normalized * 10
        
        # 6. Specialization Score (0-10 points)
        df['is_specialized'] = df['vendor_name'].str.contains('Div', case=False, na=False).astype(int)
        df['specialization_score'] = (1 - df['is_specialized']) * 10
        
        # Total Reliability Score
        df['reliability_score'] = (
            df['capacity_score'] + 
            df['lead_time_score'] + 
            df['discount_score'] + 
            df['geographic_score'] + 
            df['gst_score'] + 
            df['specialization_score']
        )
        
        self.reliability_scores = df[[
            'vendor_id', 'vendor_name', 'material_category', 'state',
            'capacity_score', 'lead_time_score', 'discount_score',
            'geographic_score', 'gst_score', 'specialization_score', 'reliability_score'
        ]]
        return self.reliability_scores
    
    def select_top_vendors_for_project(self, material_category=None, weight_tco=0.45, 
                                       weight_reliability=0.35, weight_ecosystem=0.2):
        """
        Select top 3 vendors for the project based on requirements and BOM.
        
        Args:
            material_category: Optional filter for material category
            weight_tco: Weight for TCO (including transport)
            weight_reliability: Weight for reliability (including location)
            weight_ecosystem: Weight for ecosystem
        
        Returns:
            pd.DataFrame: Top 3 recommended vendors
        """
        # Calculate scores
        if self.tco_scores is None:
            self.calculate_tco_with_bom()
        if self.reliability_scores is None:
            self.compute_reliability_scores()
        if self.ecosystem_map is None:
            self.build_ecosystem_map()
        
        # Start with all vendors or filter by category
        df = self.df.copy()
        if material_category:
            df = df[df['material_category'].str.lower() == material_category.lower()]
        
        # Merge scores
        df = df.merge(self.tco_scores[['vendor_id', 'tco_score']], on='vendor_id')
        df = df.merge(self.reliability_scores[['vendor_id', 'reliability_score']], on='vendor_id')
        
        # Calculate ecosystem score
        G = self.ecosystem_map
        betweenness_centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)
        
        ecosystem_scores = {}
        for vendor_id in df['vendor_id']:
            bc = betweenness_centrality.get(vendor_id, 0)
            dc = degree_centrality.get(vendor_id, 0)
            ecosystem_scores[vendor_id] = (bc * 0.6 + dc * 0.4) * 100
        
        df['ecosystem_score'] = df['vendor_id'].map(ecosystem_scores).fillna(0)
        
        # Normalize scores
        for score_col in ['tco_score', 'reliability_score', 'ecosystem_score']:
            min_val = df[score_col].min()
            max_val = df[score_col].max()
            if max_val > min_val:
                df[score_col] = (df[score_col] - min_val) / (max_val - min_val) * 100
            else:
                df[score_col] = 50
        
        # Calculate combined score
        df['combined_score'] = (
            weight_tco * df['tco_score'] +
            weight_reliability * df['reliability_score'] +
            weight_ecosystem * df['ecosystem_score']
        )
        
        # Get top 3
        top_3 = df.nlargest(3, 'combined_score')[
            ['vendor_id', 'vendor_name', 'material_category', 'state',
             'current_price_per_mt', 'capacity_mt_yr', 'lead_time_days',
             'bulk_discount_pct', 'gst_pct', 'tco_score', 'reliability_score', 
             'ecosystem_score', 'combined_score']
        ].reset_index(drop=True)
        
        return top_3
    
    def generate_bom_analysis_report(self):
        """
        Generate vendor analysis report based on BOM requirements.
        
        Returns:
            dict: Comprehensive analysis report
        """
        report = {
            'project_details': self.project_inputs,
            'bom_summary': {
                'total_items': len(self.bom_df) if self.bom_df is not None else 0,
                'bom_data': self.bom_df.to_dict(orient='records') if self.bom_df is not None else []
            },
            'tco_analysis': self.calculate_tco_with_bom().to_dict(orient='records'),
            'reliability_analysis': self.compute_reliability_scores().to_dict(orient='records'),
            'ecosystem_statistics': self.get_ecosystem_statistics(),
            'recommended_vendors': self.select_top_vendors_for_project().to_dict(orient='records')
        }
        return report
    
    def print_bom_analysis_summary(self):
        """Print formatted summary of BOM-based vendor analysis."""
        print("\n" + "="*100)
        print("PROJECT-BASED VENDOR ANALYSIS REPORT")
        print("="*100)
        
        # Project details
        print("\n1. PROJECT DETAILS")
        print("-" * 100)
        print(f"Project Type: {self.project_inputs.get('project_type', 'N/A')}")
        print(f"Location: {self.project_inputs.get('state', 'N/A')}, {self.project_inputs.get('region', 'N/A')}")
        print(f"Terrain: {self.project_inputs.get('terrain', 'N/A')}")
        print(f"Budget: ₹{self.project_inputs.get('total_budget', 'N/A')} Crores")
        print(f"Duration: {self.project_inputs.get('procurement_duration', 'N/A')} months")
        
        # BOM Summary
        if self.bom_df is not None:
            print("\n2. BILL OF MATERIALS SUMMARY")
            print("-" * 100)
            print(f"Total Materials: {len(self.bom_df)}")
            print("\nTop 5 Materials by Quantity:")
            bom_sorted = self.bom_df.nlargest(5, 'Estimated_Quantity')
            print(bom_sorted[['Material_Name', 'Estimated_Quantity', 'Unit', 'Priority']].to_string(index=False))
        
        # TCO Analysis
        print("\n3. TRUE COST OF OWNERSHIP (WITH TRANSPORT)")
        print("-" * 100)
        tco_df = self.calculate_tco_with_bom().sort_values('tco_score', ascending=False).head(5)
        print("Top 5 Vendors by TCO Score:")
        print(tco_df[['vendor_name', 'tco_per_mt', 'transport_cost', 'tco_score']].to_string(index=False))
        
        # Reliability Analysis
        print("\n4. VENDOR RELIABILITY SCORES")
        print("-" * 100)
        rel_df = self.compute_reliability_scores().sort_values('reliability_score', ascending=False).head(5)
        print("Top 5 Vendors by Reliability Score:")
        print(rel_df[['vendor_name', 'geographic_score', 'reliability_score']].to_string(index=False))
        
        # Ecosystem Analysis
        print("\n5. VENDOR ECOSYSTEM ANALYSIS")
        print("-" * 100)
        self.build_ecosystem_map()
        eco_stats = self.get_ecosystem_statistics()
        print(f"Total Vendors: {eco_stats['total_vendors']}")
        print(f"Total Relationships: {eco_stats['total_relationships']}")
        print(f"Average Connections: {eco_stats['avg_connections_per_vendor']:.2f}")
        print(f"Network Density: {eco_stats['network_density']:.4f}")
        
        # Final Recommendations
        print("\n6. TOP 3 RECOMMENDED VENDORS FOR PROJECT")
        print("-" * 100)
        top_vendors = self.select_top_vendors_for_project()
        print(top_vendors[['vendor_id', 'vendor_name', 'material_category', 'state',
                          'combined_score', 'tco_score', 'reliability_score']].to_string(index=False))
        
        for idx, row in top_vendors.iterrows():
            print(f"\n  VENDOR {idx+1}: {row['vendor_name']} ⭐")
            print(f"    Price: ₹{row['current_price_per_mt']:,.0f}/MT | Capacity: {row['capacity_mt_yr']:,} MT/Year")
            print(f"    Location: {row['state']} | Lead Time: {row['lead_time_days']} days")
            print(f"    Combined Score: {row['combined_score']:.2f}/100")
        
        print("\n" + "="*100)




# ============== MAIN EXECUTION ==============

def analyze_vendors_for_project(project_inputs, bom_df):
    """
    Analyze and recommend top 3 vendors based on project requirements and BOM.
    
    Args:
        project_inputs: Dict with project details (state, region, project_type, etc.)
        bom_df: DataFrame with Bill of Materials
    
    Returns:
        BOMBasedVendorAnalyzer: Analyzer with results
    """
    csv_path = 'vendors.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        return None
    
    # Initialize BOM-based analyzer
    analyzer = BOMBasedVendorAnalyzer(csv_path, project_inputs=project_inputs, bom_df=bom_df)
    
    # Print summary
    analyzer.print_bom_analysis_summary()
    
    # Build and visualize ecosystem
    analyzer.build_ecosystem_map()
    analyzer.visualize_ecosystem('vendor_ecosystem_map_project.png')
    
    # Generate report
    report = analyzer.generate_bom_analysis_report()
    
    # Save report
    with open('vendor_analysis_report_project.json', 'w') as f:
        report_json = json.dumps(
            report,
            default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x),
            indent=2
        )
        f.write(report_json)
    
    print("\n✅ Project-based vendor analysis saved to vendor_analysis_report_project.json")
    print("✅ Ecosystem visualization saved to vendor_ecosystem_map_project.png")
    
    return analyzer


def main():
    """Main function to run vendor analysis (standalone mode)."""
    csv_path = 'vendors.csv'
    analyzer = VendorAnalyzer(csv_path)
    
    # Run comprehensive analysis
    analyzer.print_analysis_summary()
    
    # Build and visualize ecosystem
    analyzer.build_ecosystem_map()
    analyzer.visualize_ecosystem('vendor_ecosystem_map.png')
    
    # Generate detailed report
    report = analyzer.generate_analysis_report()
    
    # Save report to JSON
    with open('vendor_analysis_report.json', 'w') as f:
        report_json = json.dumps(
            report,
            default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x),
            indent=2
        )
        f.write(report_json)
    
    print("\n✅ Analysis report saved to vendor_analysis_report.json")
    print("✅ Ecosystem map visualization saved to vendor_ecosystem_map.png")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()

