# v17 parallel judge manual audit sample

Scope: differential judge sanity check for parallel-family sibling arena.
Status: CONFIRMED by Ian. Judge may be frozen before s07 final evaluation.

## 1. live_parallel_15-11-0 (live_parallel)
- judge_call_success: False
- router_success: False
- parse_error: None
- predicted_calls: `[{"name": "cmd_controller.execute", "arguments": {"command": "dir C:\\ && echo Testing > C:\\testing.txt", "unit": "seconds"}}]`
- ground_truth_calls: `[{"name": "cmd_controller.execute", "accepted_arguments": {"command": ["dir c:\\", "dir c:\\\\"]}}, {"name": "cmd_controller.execute", "accepted_arguments": {"command": ["echo.>C:\\testing.txt", "echo.>C:\\\\testing.txt"], "unit": ["", "N/A"]}}]`
- raw_output_prefix: `[{"name": "cmd_controller.execute", "arguments": {"command": "dir C:\\ && echo Testing > C:\\testing.txt", "unit": "seconds"}}]`
- human_verdict: TODO

## 2. live_parallel_0-0-0 (live_parallel)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "get_current_weather", "arguments": {"location": "北京, 中国", "unit": "celsius"}}, {"name": "get_current_weather", "arguments": {"location": "上海, 中国", "unit": "celsius"}}]`
- ground_truth_calls: `[{"name": "get_current_weather", "accepted_arguments": {"location": ["Beijing, China"], "unit": ["", "fahrenheit"]}}, {"name": "get_current_weather", "accepted_arguments": {"location": ["Shanghai, China"], "unit": ["", "fahrenheit"]}}]`
- raw_output_prefix: `[{"name": "get_current_weather", "arguments": {"location": "北京, 中国", "unit": "celsius"}}, {"name": "get_current_weather", "arguments": {"location": "上海, 中国", "unit": "celsius"}}]`
- human_verdict: TODO

## 3. live_parallel_4-1-0 (live_parallel)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "get_current_weather", "arguments": {"location": "Boston, USA", "url": "https://api.open-meteo.com/v1/forecast"}}, {"name": "get_current_weather", "arguments": {"location": "San Francisco, USA", "url": "https://api.open-meteo.com/v1/forecast"}}]`
- ground_truth_calls: `[{"name": "get_current_weather", "accepted_arguments": {"location": ["Boston, USA"], "url": ["", "https://api.open-meteo.com/v1/forecast"]}}, {"name": "get_current_weather", "accepted_arguments": {"location": ["San Francisco, USA"], "url": ["", "https://api.open-meteo.com/v1/forecast"]}}]`
- raw_output_prefix: `[{"name": "get_current_weather", "arguments": {"location": "Boston, USA", "url": "https://api.open-meteo.com/v1/forecast"}}, {"name": "get_current_weather", "arguments": {"location": "San Francisco, USA", "url": "https://api.open-meteo.com/v1/forecast"}}]`
- human_verdict: TODO

## 4. live_parallel_9-5-0 (live_parallel)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "get_aws_pricing", "arguments": {"memory": 2, "cpu": "single", "region": "us-east-1", "operating_system": "Linux"}}, {"name": "get_aws_pricing", "arguments": {"memory": 4, "cpu": "single", "region": "us-east-1", "operating_system": "Linux"}}]`
- ground_truth_calls: `[{"name": "get_aws_pricing", "accepted_arguments": {"memory": [2], "cpu": ["single"], "region": ["", "us-east-1"], "operating_system": ["", "Linux"]}}, {"name": "get_aws_pricing", "accepted_arguments": {"memory": [4], "cpu": ["single"], "region": ["", "us-east-1"], "operating_system": ["", "Linux"]}}]`
- raw_output_prefix: `[{"name": "get_aws_pricing", "arguments": {"memory": 2, "cpu": "single", "region": "us-east-1", "operating_system": "Linux"}}, {"name": "get_aws_pricing", "arguments": {"memory": 4, "cpu": "single", "region": "us-east-1", "operating_system": "Linux"}}]`
- human_verdict: TODO

## 5. live_parallel_multiple_0-0-0 (live_parallel_multiple)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "ChaFod", "arguments": {"foodItem": "Caesar Salad", "removeIngredients": "anchovies", "specialInstructions": ""}}, {"name": "ChaDri.change_drink", "arguments": {"drink_id": "123", "new_preferences": {"size": "large", "temperature": "hot", "sweetness_level": "regular", "milk_type": "almond", "special_instructions": ""}}}]`
- ground_truth_calls: `[{"name": "ChaFod", "accepted_arguments": {"foodItem": ["Caesar salad"], "newIngredients": [""], "removeIngredients": ["anchovies", "anchovy"], "specialInstructions": [""]}}, {"name": "ChaDri.change_drink", "accepted_arguments": {"drink_id": ["123"], "new_preferences": [{"size": ["large"], "temperature": ["hot"], "sweetness_level": ["", "regular"], "milk_type": ["almond"], "special_instructions": [""]}]}}]`
- raw_output_prefix: `[{"name": "ChaFod", "arguments": {"foodItem": "Caesar Salad", "removeIngredients": "anchovies", "specialInstructions": ""}}, {"name": "ChaDri.change_drink", "arguments": {"drink_id": "123", "new_preferences": {"size": "large", "temperature": "hot", "sweetness_level": "regular", "milk_type": "almond", "special_instructions": ""}}}]`
- human_verdict: TODO

## 6. live_parallel_multiple_10-9-0 (live_parallel_multiple)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "recall_memory_search", "arguments": {"query": "Shishir's birthday", "page": 0}}, {"name": "send_message", "arguments": {"message": "Happy Birthday! Wishing you a wonderful day filled with joy and happiness.", "recipient": "Shishir", "urgent": false}}]`
- ground_truth_calls: `[{"name": "recall_memory_search", "accepted_arguments": {"query": ["Shishir Birthday", "Birthday Shishir", "Shishir's Birthday", "when is shishir's birthday", "when shishir birthday is", "birthday"], "page": ["", 0], "request_heartbeat": ["", false]}}, {"name": "send_message", "accepted_arguments": {"message": ["Happy Birthday!"], "recipient": ["Shishir"], "urgent": ["", false]}}]`
- raw_output_prefix: `[{"name": "recall_memory_search", "arguments": {"query": "Shishir's birthday", "page": 0}}, {"name": "send_message", "arguments": {"message": "Happy Birthday! Wishing you a wonderful day filled with joy and happiness.", "recipient": "Shishir", "urgent": false}}]`
- human_verdict: TODO

## 7. live_parallel_multiple_2-2-0 (live_parallel_multiple)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "ControlAppliance.execute", "arguments": {"command": "거실, 에어컨, 실행"}}, {"name": "ControlAppliance.execute", "arguments": {"command": "침실, 공기청정기, 중지"}}]`
- ground_truth_calls: `[{"name": "ControlAppliance.execute", "accepted_arguments": {"command": ["거실, 에어컨, 실행"]}}, {"name": "ControlAppliance.execute", "accepted_arguments": {"command": ["침실, 공기청정기, 중지"]}}]`
- raw_output_prefix: `[{"name": "ControlAppliance.execute", "arguments": {"command": "거실, 에어컨, 실행"}}, {"name": "ControlAppliance.execute", "arguments": {"command": "침실, 공기청정기, 중지"}}]`
- human_verdict: TODO

## 8. live_parallel_multiple_18-16-0 (live_parallel_multiple)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "Events_3_FindEvents", "arguments": {"event_type": "Music", "city": "San Diego, CA", "date": "2023-03-01"}}, {"name": "Events_3_FindEvents", "arguments": {"event_type": "Theater", "city": "San Diego, CA", "date": "2023-03-01"}}]`
- ground_truth_calls: `[{"name": "Events_3_FindEvents", "accepted_arguments": {"event_type": ["Music"], "city": ["San Diego, CA"], "date": ["2023-03-01"]}}, {"name": "Events_3_FindEvents", "accepted_arguments": {"event_type": ["Theater"], "city": ["San Diego, CA"], "date": ["2023-03-01"]}}]`
- raw_output_prefix: `[{"name": "Events_3_FindEvents", "arguments": {"event_type": "Music", "city": "San Diego, CA", "date": "2023-03-01"}}, {"name": "Events_3_FindEvents", "arguments": {"event_type": "Theater", "city": "San Diego, CA", "date": "2023-03-01"}}]`
- human_verdict: TODO

## 9. parallel_29 (parallel)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "waste_calculation.calculate", "arguments": {"population": {"adults": 2, "children": 2, "singles": 0}, "location": "Los Angeles"}}, {"name": "waste_calculation.calculate", "arguments": {"population": {"adults": 1, "children": 0, "singles": 1}, "location": "New York"}}]`
- ground_truth_calls: `[{"name": "waste_calculation.calculate", "accepted_arguments": {"population": [{"adults": [2], "children": [2], "singles": [0]}], "location": ["Los Angeles", "Los Angeles, CA", "LA"]}}, {"name": "waste_calculation.calculate", "accepted_arguments": {"population": [{"adults": [0], "children": [0], "singles": [1]}], "location": ["New York", "New York, NY", "NY", "New York City", "NYC"]}}]`
- raw_output_prefix: `[{"name": "waste_calculation.calculate", "arguments": {"population": {"adults": 2, "children": 2, "singles": 0}, "location": "Los Angeles"}}, {"name": "waste_calculation.calculate", "arguments": {"population": {"adults": 1, "children": 0, "singles": 1}, "location": "New York"}}]`
- human_verdict: TODO

## 10. parallel_51 (parallel)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "recipe_search.find", "arguments": {"dish": "Chicken Noodle Soup"}}, {"name": "recipe_search.find", "arguments": {"dish": "Vegan Salad", "diet": "Vegan"}}]`
- ground_truth_calls: `[{"name": "recipe_search.find", "accepted_arguments": {"dish": ["Chicken Noodle Soup"], "diet": ["", "Keto"]}}, {"name": "recipe_search.find", "accepted_arguments": {"dish": ["Salad", "salad", "Vegan Salad", "vegan salad"], "diet": ["Vegan"]}}]`
- raw_output_prefix: `[{"name": "recipe_search.find", "arguments": {"dish": "Chicken Noodle Soup"}}, {"name": "recipe_search.find", "arguments": {"dish": "Vegan Salad", "diet": "Vegan"}}]`
- human_verdict: TODO

## 11. parallel_141 (parallel)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "get_theater_movie_releases", "arguments": {"location": "New York", "timeframe": 7, "format": "IMAX"}}, {"name": "get_theater_movie_releases", "arguments": {"location": "Los Angeles", "timeframe": 14, "format": "2D"}}]`
- ground_truth_calls: `[{"name": "get_theater_movie_releases", "accepted_arguments": {"location": ["New York", "New York, NY", "NYC"], "timeframe": [7], "format": ["IMAX", ""]}}, {"name": "get_theater_movie_releases", "accepted_arguments": {"location": ["Los Angeles", "Los Angeles, CA", "LA"], "timeframe": [14], "format": ["2D"]}}]`
- raw_output_prefix: `[{"name": "get_theater_movie_releases", "arguments": {"location": "New York", "timeframe": 7, "format": "IMAX"}}, {"name": "get_theater_movie_releases", "arguments": {"location": "Los Angeles", "timeframe": 14, "format": "2D"}}]`
- human_verdict: TODO

## 12. parallel_107 (parallel)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "get_protein_sequence", "arguments": {"gene": "BRCA1", "species": "Homo sapiens"}}, {"name": "get_protein_sequence", "arguments": {"gene": "BRCA1", "species": "Pan troglodytes"}}, {"name": "get_protein_sequence", "arguments": {"gene": "BRCA2", "species": "Homo sapiens"}}, {"name": "get_protein_sequence", "arguments": {"gene": "BRCA2", "species": "Pan troglodytes"}}]`
- ground_truth_calls: `[{"name": "get_protein_sequence", "accepted_arguments": {"gene": ["BRCA1"], "species": ["Homo sapiens", ""]}}, {"name": "get_protein_sequence", "accepted_arguments": {"gene": ["BRCA2"], "species": ["Homo sapiens", ""]}}, {"name": "get_protein_sequence", "accepted_arguments": {"gene": ["BRCA1"], "species": ["Pan troglodytes"]}}, {"name": "get_protein_sequence", "accepted_arguments": {"gene": ["BRCA2"], "species": ["Pan troglodytes"]}}]`
- raw_output_prefix: `[{"name": "get_protein_sequence", "arguments": {"gene": "BRCA1", "species": "Homo sapiens"}}, {"name": "get_protein_sequence", "arguments": {"gene": "BRCA1", "species": "Pan troglodytes"}}, {"name": "get_protein_sequence", "arguments": {"gene": "BRCA2", "species": "Homo sapiens"}}, {"name": "get_protein_sequence", "arguments": {"gene": "BRCA2", "species": "Pan troglodytes"}}]`
- human_verdict: TODO

## 13. parallel_multiple_53 (parallel_multiple)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "board_game_search", "arguments": {"complexity": 2.5, "player_count": 5}}, {"name": "trivia_game_search", "arguments": {"duration": 60}}]`
- ground_truth_calls: `[{"name": "board_game_search", "accepted_arguments": {"complexity": [2.5], "player_count": [6]}}, {"name": "trivia_game_search", "accepted_arguments": {"duration": [60.0, 45.0, 30.0]}}]`
- raw_output_prefix: `[{"name": "board_game_search", "arguments": {"complexity": 2.5, "player_count": 5}}, {"name": "trivia_game_search", "arguments": {"duration": 60}}]`
- human_verdict: TODO

## 14. parallel_multiple_33 (parallel_multiple)
- judge_call_success: False
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "supermarket.find_in_city", "arguments": {"city": "Los Angeles", "state": "CA"}}, {"name": "sightseeing.popular_in_city", "arguments": {"city": "Miami", "state": "FL"}}]`
- ground_truth_calls: `[{"name": "supermarket.find_in_city", "accepted_arguments": {"city": ["Los Angeles", "LA"], "state": ["California", "CA"], "openNow": ["", true]}}, {"name": "sightseeing.popular_in_city", "accepted_arguments": {"city": ["Miami"], "state": ["Florida", "FL"], "kidsFriendly": ["", true]}}]`
- raw_output_prefix: `[{"name": "supermarket.find_in_city", "arguments": {"city": "Los Angeles", "state": "CA"}}, {"name": "sightseeing.popular_in_city", "arguments": {"city": "Miami", "state": "FL"}}]`
- human_verdict: TODO

## 15. parallel_multiple_24 (parallel_multiple)
- judge_call_success: True
- router_success: True
- parse_error: None
- predicted_calls: `[{"name": "investment.invest", "arguments": {"company": "Google", "amount": 2000.0}}, {"name": "investment.withdraw", "arguments": {"company": "Apple", "amount": 1000.0}}]`
- ground_truth_calls: `[{"name": "investment.invest", "accepted_arguments": {"company": ["Google", "GOOG"], "amount": [2000.0]}}, {"name": "investment.withdraw", "accepted_arguments": {"company": ["Apple", "AAPL"], "amount": [1000.0]}}]`
- raw_output_prefix: `[{"name": "investment.invest", "arguments": {"company": "Google", "amount": 2000.0}}, {"name": "investment.withdraw", "arguments": {"company": "Apple", "amount": 1000.0}}]`
- human_verdict: TODO
