from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# 드라이버 경로 설정 (본인 경로에 맞게 수정하세요)
driver_path = r"C:\Users\박성제\문서\Deeplearning\selenium\chromedriver-win64\chromedriver-win64\chromedriver.exe"
service = Service(driver_path)

# 크롬 드라이버 실행
driver = webdriver.Chrome(service=service)

try:
    # 1. SNU 포털 접속
    driver.get("https://my.snu.ac.kr/login.jsp")
    time.sleep(2)  # 페이지 로딩 대기

    # 2. 로그인 버튼 클릭 (로그인 창 열기)
    login_button = driver.find_element(By.XPATH, "//button[text()='로그인']")
    login_button.click()
    time.sleep(2)

    login_button = driver.find_element(By.ID, "tab-4")
    login_button.click()
    time.sleep(2)

    # 3. ID 입력
    username_input = driver.find_element(By.ID, "login_id")
    username_input.send_keys("your_id")  # 여기에 아이디 입력

    # 4. PW 입력
    password_input = driver.find_element(By.ID, "login_pwd")
    password_input.send_keys("your_password")  # 여기에 비밀번호 입력

    # 5. 최종 로그인 버튼 클릭
    submit_button = driver.find_element(By.ID, "loginProcBtn")
    submit_button.click()
    
    # 6. 로그인 결과 확인 (페이지 제목 출력)
    time.sleep(3)
    print("로그인 후 페이지 제목:", driver.title)

finally:
    # 드라이버 종료
    driver.quit()
