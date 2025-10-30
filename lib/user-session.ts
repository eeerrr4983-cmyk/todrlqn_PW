/**
 * User Session Management (LB-10)
 * 
 * Provides consistent user identification across the application
 * without requiring actual login for guest users.
 */

/**
 * Get consistent user-specific student ID
 * 
 * Priority order:
 * 1. Stored student ID (when user has provided it)
 * 2. Hash-based ID from user session (consistent per user)
 * 3. Default fallback "0000"
 * 
 * @returns 4-digit student ID string
 */
export function getUserStudentId(): string {
  if (typeof window === 'undefined') return '0000'
  
  // Try to get from sessionStorage (user's actual student ID)
  const storedStudentId = sessionStorage.getItem('student_id')
  if (storedStudentId && storedStudentId.length === 4) {
    return storedStudentId
  }
  
  // Generate consistent ID based on user session (NOT per analysis)
  const userSessionId = sessionStorage.getItem('user_session_id') || ''
  if (userSessionId) {
    // Hash the session ID to get consistent 4 digits for this user
    let hash = 0
    for (let i = 0; i < userSessionId.length; i++) {
      hash = ((hash << 5) - hash) + userSessionId.charCodeAt(i)
      hash = hash & hash // Convert to 32bit integer
    }
    const fourDigits = String(Math.abs(hash) % 10000).padStart(4, '0')
    return fourDigits
  }
  
  return '0000'
}

/**
 * Get user display name for UI
 * 
 * @returns Display name (e.g., "Student67" for anonymous or real name when available)
 */
export function getUserDisplayName(): string {
  if (typeof window === 'undefined') return '사용자'
  
  // Check if user has provided a name
  const storedName = sessionStorage.getItem('student_name')
  if (storedName && storedName.trim()) {
    return storedName
  }
  
  // Use "Student + last 2 digits" format for anonymous users
  const studentId = getUserStudentId()
  const lastTwoDigits = studentId.slice(-2)
  return `Student${lastTwoDigits}`
}

/**
 * Check if current user is the owner of an analysis
 * 
 * @param analysisStudentId - Student ID from the analysis record
 * @returns true if the current user owns this analysis
 */
export function isCurrentUser(analysisStudentId: string): boolean {
  const currentStudentId = getUserStudentId()
  return currentStudentId === analysisStudentId
}

/**
 * Format student ID for display
 * 
 * @param studentId - 4-digit student ID
 * @param showAsMe - If true and it's the current user, return "나"
 * @returns Formatted string (Student67 format for anonymous users)
 */
export function formatStudentId(studentId: string, showAsMe: boolean = false): string {
  if (showAsMe && isCurrentUser(studentId)) {
    return '나'
  }
  // Use "Student + last 2 digits" format
  const lastTwoDigits = studentId.slice(-2)
  return `Student${lastTwoDigits}`
}

/**
 * Initialize or get user session ID
 * Should be called on app initialization
 * 
 * @returns User session ID
 */
export function initUserSession(): string {
  if (typeof window === 'undefined') return ''
  
  let sessionId = sessionStorage.getItem('user_session_id')
  
  if (!sessionId) {
    // Generate new session ID
    sessionId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    sessionStorage.setItem('user_session_id', sessionId)
  }
  
  return sessionId
}

/**
 * Clear user session (for logout or reset)
 */
export function clearUserSession(): void {
  if (typeof window === 'undefined') return
  
  sessionStorage.removeItem('user_session_id')
  sessionStorage.removeItem('student_id')
  sessionStorage.removeItem('student_name')
  sessionStorage.removeItem('current_analysis')
  sessionStorage.removeItem('is_analyzing')
}
