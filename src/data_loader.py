import ast
from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoader:
    """Product data loading and preprocessing."""
    
    @staticmethod
    def _parse_details(details_value):
        """Parse details dict to key:value pairs.
        
        Args:
            details_value: String like "{'Brand': 'Sony', 'Color': 'Black'}"
            
        Returns:
            Space-separated key:value pairs like "Brand:Sony Color:Black"
        """
        if pd.isna(details_value) or details_value == '':
            return ''
        try:
            if isinstance(details_value, str):
                details_dict = ast.literal_eval(details_value)
                if isinstance(details_dict, dict):
                    return ' '.join(f"{k}:{v}" for k, v in details_dict.items() if v)
            return str(details_value)
        except:
            return str(details_value)
    
    @staticmethod
    def _parse_categories(categories_value):
        """Parse categories list to space-separated string.
        
        Args:
            categories_value: String like "['Electronics', 'Audio', 'Headphones']"
            
        Returns:
            Space-separated categories "Electronics Audio Headphones"
        """
        if pd.isna(categories_value) or categories_value == '':
            return ''
        try:
            if isinstance(categories_value, str):
                categories_list = ast.literal_eval(categories_value)
                if isinstance(categories_list, (list, tuple)):
                    return ' '.join(str(cat) for cat in categories_list if cat)
            return str(categories_value)
        except:
            return str(categories_value)
    
    def load_data(self, data_path: Path, max_products: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare product data with caching.
        
        Args:
            data_path: Path to CSV with product reviews and metadata
            max_products: Limit number of unique products (for testing)
            
        Returns:
            DataFrame with parent_asin, title, category, ratings, text_features
        """
        cache_path = Path(data_path).parent / f'processed_products_{max_products if max_products else "all"}.pkl'
        
        if cache_path.exists():
            product_df = pd.read_pickle(cache_path)
            print(f"Loaded {len(product_df):,} products from cache")
            return product_df
        
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} products")
        
        df['details_parsed'] = df['details'].apply(self._parse_details)
        df['categories_parsed'] = df['categories'].apply(self._parse_categories)
        
        df['text_features'] = (
            df['title'].fillna('') + ' ' + 
            df['details_parsed'].fillna('') + ' ' +
            df['categories_parsed'].fillna('')
        ).astype(str)
        
        product_df = df.groupby('parent_asin').agg({
            'title': 'first',
            'main_category': 'first',
            'average_rating': 'first',
            'rating_number': 'first',
            'text_features': 'first',
            'categories': 'first'
        }).reset_index()
        
        product_df['text_features'] = product_df['text_features'].astype(str)
        
        if max_products and len(product_df) > max_products:
            product_df = product_df.sample(n=max_products, random_state=42).reset_index(drop=True)
        
        print(f"Prepared {len(product_df):,} products")
        return product_df
