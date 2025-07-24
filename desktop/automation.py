import subprocess
import difflib




def find_best_app_match(query, apps):

    query_lower = query.lower()
    app_names_lower = [app["name"].lower() for app in apps]
    

    matches = difflib.get_close_matches(query_lower, app_names_lower, n=1, cutoff=0.4)
    if matches:

        return next(app for app in apps if app["name"].lower() == matches[0])
    

    for app in apps:
        if query_lower in app["name"].lower():
            return app
    return None

def main(translate):

    try:
        check_apps = subprocess.check_output(["powershell", "Get-StartApps"], text=True)

        apps = []
        for line in check_apps.splitlines():
            line = line.strip()

            if not line or "AppUserModelId" in line:
                continue

            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            display_name, app_id = parts[0], parts[1]
            apps.append({"name": display_name, "id": app_id})

        if not apps:
            print("No apps found.")
            exit(1)


        matched_app = find_best_app_match(translate, apps)

        if matched_app:
            corrected_name = matched_app["name"]
            if corrected_name.lower() != translate.lower():
                print(f"Correcting '{translate}' to '{corrected_name}'.")
            else:
                print(f"Found app '{corrected_name}'.")
                # voice(f"Opening {corrected_name}")
            try:

                cmd = f'Start-Process "shell:AppsFolder\\{matched_app["id"]}"'
                subprocess.run(["powershell", "-Command", cmd], check=True)
                print(f"Launched {corrected_name}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error launching app: {e}")
        else:
            print(f"Could not find an app matching '{translate}'.")

    except subprocess.CalledProcessError as e:
        print(f"Error running Get-StartApps: {e}")


if __name__ == "__main__":
    main("open netflix")