from typing import List, Dict


class RecommendationDisplay:
    """Handles displaying product recommendations in a formatted output.
    
    This class provides methods to display query products and their recommendations
    in a readable, formatted console output.
    """
    
    def display_recommendations(self, query_product: Dict, similar_products: List[Dict], method: str):
        """Display product recommendations in a formatted output.
        
        Args:
            query_product: Dictionary containing query product information
            similar_products: List of similar products with metadata and scores
            method: Name of the recommendation method used
        """
        print(f"\n{'='*80}")
        print(f"Method: {method}")
        print(f"{'='*80}")
        print(f"\nQuery Product:")
        print(f"  ASIN:     {query_product['parent_asin']}")
        print(f"  Title:    {query_product['title'][:80]}...")
        print(f"  Category: {query_product['main_category']}")
        print(f"  Rating:   {query_product['average_rating']:.1f} ({query_product['rating_number']} reviews)")
        print(f"\nTop {len(similar_products)} Similar Products:")
        
        for prod in similar_products:
            print(f"\n  {prod['rank']}. Similarity: {prod['similarity']:.4f}")
            print(f"     ASIN:     {prod['asin']}")
            print(f"     Title:    {prod['title'][:80]}...")
            print(f"     Category: {prod['category']}")
            print(f"     Rating:   {prod['rating']:.1f} ({prod['rating_number']} reviews)")
