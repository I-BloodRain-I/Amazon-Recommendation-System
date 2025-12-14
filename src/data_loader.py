import ast
from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoader:
    """Handles loading and preprocessing of product data.
    
    This class provides methods to load product data from CSV files and preprocess
    it by parsing details and categories columns into usable text features.
    """
    
    @staticmethod
    def _parse_details(details_value):
        """Parse details column from string representation of dict to key:value pairs.
        
        Args:
            details_value: String representation of a dictionary or raw value
            
        Returns:
            Formatted string with key:value pairs or empty string on error
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
        """Parse categories column from string representation of list/array to actual values.
        
        Args:
            categories_value: String representation of a list/tuple or raw value
            
        Returns:
            Space-separated string of categories or empty string on error
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
        """Load and prepare product data from CSV file with caching support.
        
        Args:
            data_path: Path to the CSV file containing product data
            max_products: Optional maximum number of products to load (for sampling)
            
        Returns:
            DataFrame with processed product data including text features
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
