from datetime import datetime

def calc_Costs(injection_dates, withdrawal_dates, purchase_prices, sale_prices, injection_rate, withdrawal_rate, max_volume, storage_costs):
    # Convert strings to datetime objects with "MM/DD/YYYY"
    injection_dates = [datetime.strptime(date, "%m/%d/%Y") for date in injection_dates]
    withdrawal_dates = [datetime.strptime(date, "%m/%d/%Y") for date in withdrawal_dates]
    
    # First iteration: Calculate total storage duration in days for all periods
    total_storage_days = sum([(withdrawal_dates[i] - injection_dates[i]).days for i in range(len(injection_dates))])
    
    # Calculate average purchase and sale prices if multiple periods are given
    avg_purchase_price = sum(purchase_prices) / len(purchase_prices)
    avg_sale_price = sum(sale_prices) / len(sale_prices)
    
    # Calculate revenue from sale and cost of purchase
    revenue_from_sale = avg_sale_price * max_volume
    cost_of_purchase = avg_purchase_price * max_volume
    
    # Calculate total storage cost
    total_storage_cost = (total_storage_days / 30.4) * storage_costs  # Assuming storage_costs are monthly
    
    # Calculate total injection and withdrawal cost
    total_injection_withdrawal_cost = (injection_rate + withdrawal_rate) * max_volume//1e6  # Assuming rates are per 1 million MMBtu
    
    # Calculate total transport cost (fixed $50 each time for both injection and withdrawal)
    total_transport_cost = 2 * 50 * len(injection_dates)
    
    # Total value calculation
    total_value = revenue_from_sale - cost_of_purchase - total_storage_cost - total_injection_withdrawal_cost - total_transport_cost
    
    return total_value

# Test and print section
def main():
    # Sample inputs
    injection_dates = ["03/01/2023"]
    withdrawal_dates = ["12/01/2025"]
    purchase_prices = [2]  # $2/MMBtu
    sale_prices = [6]  # $3/MMBtu
    injection_rate = 10  # $10K per 1 million MMBtu for injection
    withdrawal_rate = 10  # $10K per 1 million MMBtu for withdrawal
    max_volume = 1e6  # 1 million MMBtu
    storage_costs = 1e5  # $100K a month

    # Calculate the contract value
    contract_value = calc_Costs(injection_dates, withdrawal_dates, purchase_prices, sale_prices, injection_rate, withdrawal_rate, max_volume, storage_costs)
    print(f"The value of the contract is: ${contract_value:.2f}")


main()
