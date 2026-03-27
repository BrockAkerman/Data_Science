from librarian import get_query_engine

def start_consultation():
    print("\n" + "="*50)
    print("DS TOOLBOX LIBRARIAN IS ONLINE")
    print("Type 'exit' to quit.")
    print("="*50)

    # Initialize the engine once
    engine = get_query_engine()

    while True:
        user_input = input("\nHow can I help with your DS project? > ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Closing Librarian. Happy coding!")
            break
            
        if not user_input.strip():
            continue

        # Update this part in chat.py
        response = engine.query(user_input)
        print("\n[LIBRARIAN]:")
        response.print_response_stream()
        print("\n") # Add a newline at the end

if __name__ == "__main__":
    start_consultation()